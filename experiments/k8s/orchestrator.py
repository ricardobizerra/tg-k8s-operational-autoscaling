import argparse
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import urlopen
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from jinja2 import Template

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
  datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


MANIFESTS_DIR = Path(__file__).parent / "manifests"

SLUG_MAP = {
  "ExponentialSmoothing": "expsm",
  "Prophet": "prophet",
  "FFT": "fft",
  "GRU": "gru",
  "Transformer": "transformer",
  "BlockRNNGRU": "blockrnngru",
}

MODE_SUFFIX = {
  "horizontal": "honly",
  "vertical": "vonly",
  "combined": "hv",
}

ALL_MODELS = ["ExponentialSmoothing", "Prophet", "FFT", "Ensemble"]
ALL_DATASETS = ["a-cpu", "a-ram", "b-cpu", "b-ram", "c-cpu", "c-ram"]
ALL_MODES = ["horizontal", "vertical", "combined"]

DEFAULT_ENSEMBLE_MODELS = ["ExponentialSmoothing", "Prophet", "FFT"]

PORT_POOL_MIN = 30100
PORT_POOL_MAX = 30199

CPU_LIMIT_CORES = "2"
CPU_TARGET_PER_REPLICA = "0.5"
INITIAL_MEMORY_MB = 512
MAX_MEMORY_MB = 8192
MEMORY_STEP_MB = 1024
MIN_REPLICAS = 1
MAX_REPLICAS = 10
PEAK_RPS = 27


@dataclass
class ExperimentSpec:
  model_name: str
  ensemble_models: List[str]
  dataset_slug: str
  scaling_mode: str
  model_slug: str = field(init=False)
  namespace_name: str = field(init=False)

  def __post_init__(self):
    self.model_slug = _slug_model(self.model_name, self.ensemble_models)
    self.namespace_name = _namespace_name_from_parts(
      self.model_slug, self.dataset_slug, self.scaling_mode
    )


def _slug_model(model_name: str, ensemble_models: List[str]) -> str:
  if model_name == "Ensemble":
    parts = [SLUG_MAP.get(m, m.lower()) for m in ensemble_models]
    return "ens-" + "-".join(parts)
  return SLUG_MAP.get(model_name, model_name.lower())


def _namespace_name_from_parts(model_slug: str, dataset_slug: str, scaling_mode: str) -> str:
  suffix = MODE_SUFFIX[scaling_mode]
  return f"tcc-{model_slug}-{dataset_slug}-{suffix}"


def _model_save_name(model_name: str, ensemble_models: List[str]) -> str:
  if model_name == "Ensemble" and ensemble_models:
    return f"Ensemble[{'+'.join(ensemble_models)}]"
  return model_name


def _ram_slug(dataset_slug: str) -> str:
  app = dataset_slug.split("-")[0]
  return f"{app}-ram"


def _parse_duration(duration_str: str) -> int:
  m = re.fullmatch(r"(\d+)([dDsS]?)", duration_str.strip())
  if not m:
    raise argparse.ArgumentTypeError(f"Invalid duration: {duration_str}")
  value, unit = int(m.group(1)), m.group(2).lower()
  if unit == "d":
    return value * 86400
  return value  # default unit: seconds


def _metric_trace_step_seconds(args, dataset_slug: str) -> int:
  metric = dataset_slug.split("-", 1)[1]
  if metric == "cpu" and getattr(args, "cpu_trace_step_seconds", None) is not None:
    return args.cpu_trace_step_seconds
  if metric == "ram" and getattr(args, "ram_trace_step_seconds", None) is not None:
    return args.ram_trace_step_seconds
  return args.trace_step_seconds


def _build_pending(
  models: List[str],
  datasets: List[str],
  scaling_modes: List[str],
  ensemble_models: Optional[List[str]] = None,
) -> List[ExperimentSpec]:
  specs = []
  
  if "Ensemble" in models:
    if ensemble_models:
      ensemble_combinations = [sorted(ensemble_models)]
    else:
      base_models = sorted([m for m in models if m != "Ensemble"])
      if not base_models:
        base_models = sorted(DEFAULT_ENSEMBLE_MODELS)
        
      ensemble_combinations = [base_models]
  else:
    ensemble_combinations = []

  for model in models:
    for dataset in datasets:
      for mode in scaling_modes:
        if mode == "combined" and dataset.endswith("-ram"):
          continue

        if model == "Ensemble":
          for ens_combo in ensemble_combinations:
            specs.append(ExperimentSpec(
              model_name=model,
              ensemble_models=ens_combo,
              dataset_slug=dataset,
              scaling_mode=mode,
            ))
        else:
          specs.append(ExperimentSpec(
            model_name=model,
            ensemble_models=[],
            dataset_slug=dataset,
            scaling_mode=mode,
          ))
  return specs


def _load_state(state_file: Path) -> dict:
  with open(state_file) as f:
    return json.load(f)


def _save_state(state: dict, state_file: Path) -> None:
  state_file.parent.mkdir(parents=True, exist_ok=True)
  state["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
  tmp = state_file.with_suffix(".tmp")
  with open(tmp, "w") as f:
    json.dump(state, f, indent=2, default=str)
  tmp.replace(state_file)


def _init_state(args, specs: List[ExperimentSpec]) -> dict:
  config = {
    "models": args.models,
    "datasets": args.datasets,
    "scaling_modes": args.scaling_modes,
    "trace_step_seconds": args.trace_step_seconds,
    "trace_days": args.duration // 86400,
    "batch_size": args.batch_size,
  }
  if getattr(args, "cpu_trace_step_seconds", None) is not None:
    config["cpu_trace_step_seconds"] = args.cpu_trace_step_seconds
  if getattr(args, "ram_trace_step_seconds", None) is not None:
    config["ram_trace_step_seconds"] = args.ram_trace_step_seconds
  return {
    "version": 1,
    "config": config,
    "completed": [],
    "failed": [],
    "pending": [s.namespace_name for s in specs],
    "started_at": datetime.now(tz=timezone.utc).isoformat(),
    "last_updated": datetime.now(tz=timezone.utc).isoformat(),
  }


def _load_template(name: str) -> Template:
  path = MANIFESTS_DIR / f"{name}.yaml.j2"
  return Template(path.read_text())


def _build_scaler_env(spec: ExperimentSpec, args) -> Dict[str, str]:
  trace_step_seconds = _metric_trace_step_seconds(args, spec.dataset_slug)
  return {
    "NAMESPACE": spec.namespace_name,
    "DEPLOYMENT_NAME": "target-app",
    "MODEL_NAME": spec.model_name,
    "ENSEMBLE_MODELS": ",".join(spec.ensemble_models),
    "DATASET_SLUG": spec.dataset_slug,
    "SCALING_MODE": spec.scaling_mode,
    "EXP1_RESULTS_DIR": "/app/experiments/results",
    "PROMETHEUS_URL": "http://prometheus.tcc-infra.svc.cluster.local:9090",
    "MAX_HISTORY_POINTS": "500",
    "SCRAPE_INTERVAL_SECONDS": str(trace_step_seconds),
    "REFIT_TIMEOUT_SECONDS": "120",
    "MIN_REPLICAS": str(MIN_REPLICAS),
    "MAX_REPLICAS": str(MAX_REPLICAS),
    "CPU_TARGET_PER_REPLICA": CPU_TARGET_PER_REPLICA,
    "MIN_MEMORY_MB": str(INITIAL_MEMORY_MB),
    "MAX_MEMORY_MB": str(MAX_MEMORY_MB),
    "MEMORY_STEP_MB": str(MEMORY_STEP_MB),
    "WARMUP_CYCLES": "0" if getattr(args, "skip_warmup", False) else str(168 if "cpu" in spec.dataset_slug else 252),
  }


def _render_templates(spec: ExperimentSpec, node_port: int, args) -> Dict[str, str]:
  ctx = {
    "namespace_name": spec.namespace_name,
    "initial_memory_mb": INITIAL_MEMORY_MB,
    "cpu_limit_cores": CPU_LIMIT_CORES,
    "max_memory_mb": MAX_MEMORY_MB,
    "node_port": node_port,
    "scaler_env": _build_scaler_env(spec, args),
    "exp1_results_host_path": str(Path(args.exp1_results_dir).resolve()),
  }
  return {
    "namespace": _load_template("namespace").render(**ctx),
    "rbac":      _load_template("rbac").render(**ctx),
    "app":       _load_template("app").render(**ctx),
    "scaler":    _load_template("scaler").render(**ctx),
  }


def _kubectl_apply(yaml_str: str, dry_run: bool = False) -> bool:
  cmd = ["kubectl", "apply", "-f", "-"]
  if dry_run:
    cmd += ["--dry-run=client"]
  try:
    result = subprocess.run(
      cmd,
      input=yaml_str.encode(),
      capture_output=True,
      timeout=60,
    )
    if result.returncode != 0:
      logger.error("kubectl apply failed:\n%s", result.stderr.decode())
      return False
    logger.debug("kubectl apply:\n%s", result.stdout.decode())
    return True
  except subprocess.TimeoutExpired:
    logger.error("kubectl apply timed out")
    return False


def _kubectl_delete_namespace(ns: str) -> None:
  subprocess.run(
    ["kubectl", "delete", "namespace", ns, "--ignore-not-found=true"],
    capture_output=True,
    timeout=60,
  )
  logger.info("Deleted namespace: %s", ns)


def _wait_pods_ready(namespaces: List[str], timeout_s: int = 120) -> Dict[str, bool]:
  results = {}
  for ns in namespaces:
    ok = True
    for label in ("app=target-app", "app=scaler"):
      r = subprocess.run(
        ["kubectl", "wait", "--for=condition=Ready",
         "pod", "-l", label, "-n", ns,
         f"--timeout={timeout_s}s"],
        capture_output=True,
      )
      if r.returncode != 0:
        logger.error(
          "Pods not ready in %s (label=%s): %s",
          ns, label, r.stderr.decode(),
        )
        ok = False
        break
    results[ns] = ok
  return results


def _send_sigterm_to_scaler_pod(namespace: str) -> None:
  r = subprocess.run(
    ["kubectl", "exec", "-n", namespace,
     "deploy/scaler", "--", "kill", "-SIGTERM", "1"],
    capture_output=True,
    timeout=10,
  )
  if r.returncode != 0:
    logger.warning(
      "SIGTERM to scaler in %s failed (may already be gone): %s",
      namespace, r.stderr.decode(),
    )


def _compute_start_index(namespace: str, output_dir: Path, step_seconds: int, dataset_slug: str) -> int:
  scaling_csv = output_dir / namespace / "scaling_events.csv"
  warmup_csv = output_dir / namespace / "warmup_events.csv"

  if scaling_csv.exists() and os.path.getsize(scaling_csv) > 0:
    try:
      df = pd.read_csv(scaling_csv)
      if not df.empty:
        warmup_cycles = 168 if "cpu" in dataset_slug else 252
        return warmup_cycles + len(df)
    except Exception:
      pass

  if warmup_csv.exists() and os.path.getsize(warmup_csv) > 0:
    try:
      df = pd.read_csv(warmup_csv)
      return len(df)
    except Exception:
      pass

  return 0


def _start_locust(
  spec: ExperimentSpec,
  node_ip: str,
  node_port: int,
  trace_dir: Path,
  step_seconds: int,
  duration_seconds: int,
  start_index: int,
  locustfile: Path,
  output_dir: Path,
) -> subprocess.Popen:
  trace_file = trace_dir / f"{spec.dataset_slug}_trace.json"
  
  import json
  with open(trace_file) as f:
    trace_data = json.load(f)
  trace_peak_rps = max(d.get("rps", 0) for d in trace_data) if trace_data else PEAK_RPS

  users = max(trace_peak_rps * 2, PEAK_RPS * 2)
  spawn_rate = max(trace_peak_rps, PEAK_RPS)

  env = {
    **os.environ,
    "TRACE_FILE": str(trace_file),
    "TRACE_STEP_SECONDS": str(step_seconds),
    "TRACE_START_INDEX": str(start_index),
  }
  cmd = [
    sys.executable, "-m", "locust",
    "-f", str(locustfile),
    "--host", f"http://{node_ip}:{node_port}",
    "--users", str(users),
    "--spawn-rate", str(spawn_rate),
    "--headless",
    "--run-time", f"{duration_seconds}s",
  ]
  ns_dir = output_dir / spec.namespace_name
  ns_dir.mkdir(parents=True, exist_ok=True)
  log_file = open(ns_dir / "locust.log", "w")
  logger.info("[%s] Starting Locust: %s", spec.namespace_name, " ".join(cmd))
  return subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def _start_ram_injector(
  spec, node_ip: str, node_port: int,
  trace_dir: Path, step_seconds: int, start_index: int,
  injector_script: Path,
  output_dir: Path,
  max_mb: int = None,
) -> subprocess.Popen:
  ram_trace = trace_dir / f"{_ram_slug(spec.dataset_slug)}_trace.json"
  cmd = [
    sys.executable, str(injector_script),
    "--trace", str(ram_trace),
    "--host", f"http://{node_ip}:{node_port}",
    "--step-seconds", str(step_seconds),
    "--start-index", str(start_index),
    "--namespace", spec.namespace_name,
  ]
  if max_mb is not None:
    cmd.extend(["--max-mb", str(max_mb)])
  ns_dir = output_dir / spec.namespace_name
  ns_dir.mkdir(parents=True, exist_ok=True)
  log_file = open(ns_dir / "ram_injector.log", "w")
  logger.info("[%s] Starting RAM injector: %s", spec.namespace_name, " ".join(cmd))
  return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


def _count_csv_rows(namespace: str, hostpath_base: Path) -> int:
  csv_path = hostpath_base / namespace / "scaling_events.csv"
  try:
    if not csv_path.exists():
      return 0
    df = pd.read_csv(csv_path)
    return len(df)
  except Exception:
    return 0


def _pod_running(namespace: str, label: str) -> bool:
  r = subprocess.run(
    ["kubectl", "get", "pod", "-l", label, "-n", namespace,
     "--field-selector=status.phase=Running", "--no-headers"],
    capture_output=True,
  )
  return bool(r.stdout.strip())


def _wait_for_port_forward_health(local_port: int, timeout_s: int = 30) -> bool:
  deadline = time.monotonic() + timeout_s
  url = f"http://127.0.0.1:{local_port}/health"
  while time.monotonic() < deadline:
    try:
      with urlopen(url, timeout=2) as resp:
        if getattr(resp, "status", 200) == 200:
          return True
    except (URLError, OSError, TimeoutError):
      time.sleep(0.5)
  return False


def _restart_port_forward(
  ns: str,
  node_port: int,
  ns_dir: Path,
) -> subprocess.Popen:
  pf_log = open(ns_dir / "pf.log", "a")
  proc = subprocess.Popen(
    ["kubectl", "port-forward", "svc/target-app", f"{node_port}:8000", "-n", ns],
    stdout=pf_log,
    stderr=subprocess.STDOUT,
  )
  return proc


def _monitor_batch(
  specs: List[ExperimentSpec],
  locust_procs: Dict[str, subprocess.Popen],
  injector_procs: Dict[str, subprocess.Popen],
  duration_seconds: int,
  pf_procs: Dict[str, subprocess.Popen] = None,
  port_map: Dict[str, int] = None,
  output_dir: Path = None,
  interval_s: int = 60,
  node_ip: str = "127.0.0.1",
) -> None:
  deadline = time.monotonic() + duration_seconds
  while time.monotonic() < deadline:
    remaining = int(deadline - time.monotonic())

    if node_ip == "127.0.0.1" and pf_procs is not None and port_map is not None and output_dir is not None:
      for spec in specs:
        ns = spec.namespace_name
        if ns not in pf_procs:
          continue
        pf = pf_procs[ns]
        if pf.poll() is not None:  # process has exited
          logger.warning(
            "[%s] port-forward exited (code=%s) — waiting for new pod then restarting",
            ns, pf.poll(),
          )
          ready = _wait_pods_ready([ns], timeout_s=120)
          if ready.get(ns, False):
            node_port = port_map[ns]
            ns_dir = output_dir / ns
            new_pf = _restart_port_forward(ns, node_port, ns_dir)
            pf_procs[ns] = new_pf
            if _wait_for_port_forward_health(node_port, timeout_s=30):
              logger.info("[%s] port-forward restarted on :%d", ns, node_port)
            else:
              logger.error("[%s] port-forward restart health-check failed", ns)
          else:
            logger.error("[%s] new pod never became Ready after rolling restart", ns)
        else:
          node_port = port_map[ns]
          try:
            urlopen(f"http://127.0.0.1:{node_port}/health", timeout=2)
          except Exception as exc:
            logger.warning(
              "[%s] port-forward health ping failed (zombie process detected: %s). Force restarting...",
              ns, exc
            )
            pf.terminate()
            try:
              pf.wait(timeout=2)
            except subprocess.TimeoutExpired:
              pf.kill()

            ns_dir = output_dir / ns
            new_pf = _restart_port_forward(ns, node_port, ns_dir)
            pf_procs[ns] = new_pf
            if _wait_for_port_forward_health(node_port, timeout_s=30):
              logger.info("[%s] zombie port-forward successfully restarted on :%d", ns, node_port)
            else:
              logger.error("[%s] zombie port-forward restart health-check failed", ns)
    for spec in specs:
      ns = spec.namespace_name
      rows = _count_csv_rows(ns, Path("/tmp/tcc-results"))
      loc_alive = (
        ns in locust_procs and locust_procs[ns].poll() is None
      )
      inj_alive = (
        ns in injector_procs and injector_procs[ns].poll() is None
      )
      pf_alive = (
        pf_procs is None or
        (ns in pf_procs and pf_procs[ns].poll() is None)
      )
      scaler_ok = _pod_running(ns, "app=scaler")
      logger.info(
        "[%s] rows=%d locust=%s injector=%s pf=%s scaler_pod=%s remaining=%ds",
        ns, rows,
        "running" if loc_alive else "stopped",
        "running" if inj_alive else "stopped",
        "ok" if pf_alive else "RESTARTING",
        "ok" if scaler_ok else "GONE",
        remaining,
      )
    time.sleep(min(interval_s, max(0, remaining)))


def run_batch(
  specs: List[ExperimentSpec],
  args,
  state: dict,
  port_pool: deque,
  dry_run: bool = False,
) -> Tuple[List[str], List[str]]:
  completed = []
  failed = []

  port_map: Dict[str, int] = {}
  for spec in specs:
    port_map[spec.namespace_name] = port_pool.popleft()

  applied: List[ExperimentSpec] = []
  for spec in specs:
    ns = spec.namespace_name
    node_port = port_map[ns]

    results_dir = Path(f"/tmp/tcc-results/{ns}")
    results_dir.mkdir(parents=True, exist_ok=True)

    rendered = _render_templates(spec, node_port, args)

    if dry_run:
      for key, yaml_str in rendered.items():
        print(f"\n# --- {ns} / {key} ---\n{yaml_str}")
      applied.append(spec)
      continue

    ok = True
    for resource in ("namespace", "rbac", "app", "scaler"):
      if not _kubectl_apply(rendered[resource]):
        logger.error("[%s] Failed to apply %s manifest", ns, resource)
        ok = False
        break

    if ok:
      applied.append(spec)
    else:
      failed.append(ns)
      port_pool.append(node_port)

  if dry_run:
    for spec in specs:
      port_pool.append(port_map[spec.namespace_name])
    return [s.namespace_name for s in specs], []

  if not applied:
    return [], failed

  locust_procs: Dict[str, subprocess.Popen] = {}
  injector_procs: Dict[str, subprocess.Popen] = {}
  pf_procs: Dict[str, subprocess.Popen] = {}
  
  try:
    ready_map = _wait_pods_ready([s.namespace_name for s in applied], timeout_s=120)
    ready_specs = []
    for spec in applied:
      ns = spec.namespace_name
      if not ready_map.get(ns, False):
        logger.error("[%s] Pods not ready — marking as failed", ns)
        failed.append(ns)
        _kubectl_delete_namespace(ns)
        port_pool.append(port_map[ns])
      else:
        ready_specs.append(spec)

    if not ready_specs:
      return [], failed

    for spec in ready_specs:
      ns = spec.namespace_name
      mem_patch = {
        "spec": {
          "template": {
            "spec": {
              "containers": [{
                "name": "target-app",
                "resources": {
                  "limits": {"memory": f"{INITIAL_MEMORY_MB}Mi"},
                  "requests": {"memory": f"{int(INITIAL_MEMORY_MB * 0.8)}Mi"},
                },
              }]
            }
          }
        }
      }
      reset_result = subprocess.run(
        ["kubectl", "patch", "deployment", "target-app", "-n", ns,
         "--type=strategic", f"--patch={json.dumps(mem_patch)}"],
        capture_output=True, text=True,
      )
      if reset_result.returncode == 0:
        logger.info("[%s] Memory reset to %dMi", ns, INITIAL_MEMORY_MB)
      else:
        logger.warning(
          "[%s] Memory reset failed (non-fatal): %s",
          ns, reset_result.stderr.strip(),
        )

    ready_map2 = _wait_pods_ready([s.namespace_name for s in ready_specs], timeout_s=120)
    ready_specs = [s for s in ready_specs if ready_map2.get(s.namespace_name, False)]

    load_dir = Path(__file__).parent / "load"
    trace_dir = Path(args.trace_dir)
    max_duration_seconds = 0

    for spec in ready_specs:
      ns = spec.namespace_name
      node_port = port_map[ns]
      output_dir = Path(args.output_dir)
      trace_step_seconds = _metric_trace_step_seconds(args, spec.dataset_slug)
      start_idx = _compute_start_index(ns, output_dir, trace_step_seconds, spec.dataset_slug)
      
      if getattr(args, "skip_warmup", False) and start_idx == 0:
        start_idx = 168 if "cpu" in spec.dataset_slug else 252

      trace_path = trace_dir / f"{spec.dataset_slug}_trace.json"
      with open(trace_path) as f:
        trace_len = len(json.load(f))
      
      remaining_steps = trace_len - start_idx
      run_duration = remaining_steps * trace_step_seconds
      if args.duration > 0:
        run_duration = min(run_duration, args.duration)
      max_duration_seconds = max(max_duration_seconds, run_duration)

      if args.node_ip == "127.0.0.1":
        ns_dir = output_dir / ns
        ns_dir.mkdir(parents=True, exist_ok=True)
        pf_log = open(ns_dir / "pf.log", "w")
        pf_procs[ns] = subprocess.Popen(
          ["kubectl", "port-forward", "svc/target-app", f"{node_port}:8000", "-n", ns],
          stdout=pf_log,
          stderr=subprocess.STDOUT,
        )
        if not _wait_for_port_forward_health(node_port, timeout_s=30):
          logger.error(
            "[%s] port-forward never became healthy on 127.0.0.1:%d",
            ns,
            node_port,
          )
          failed.append(ns)
          if ns in pf_procs and pf_procs[ns].poll() is None:
            pf_procs[ns].terminate()
          continue

      locust_procs[ns] = _start_locust(
        spec, args.node_ip, node_port,
        trace_dir, trace_step_seconds,
        run_duration, start_idx,
        load_dir / "locustfile.py",
        output_dir,
      )

      injector_procs[ns] = _start_ram_injector(
        spec, args.node_ip, node_port,
        trace_dir, trace_step_seconds,
        start_idx,
        load_dir / "ram_injector.py",
        output_dir,
      )

    _monitor_batch(
      ready_specs, locust_procs, injector_procs, max_duration_seconds,
      pf_procs=pf_procs,
      port_map=port_map,
      output_dir=Path(args.output_dir),
      interval_s=20,
      node_ip=args.node_ip,
    )

    for spec in ready_specs:
      ns = spec.namespace_name
      _send_sigterm_to_scaler_pod(ns)
      if ns in locust_procs and locust_procs[ns].poll() is None:
        locust_procs[ns].terminate()
      if ns in injector_procs and injector_procs[ns].poll() is None:
        injector_procs[ns].terminate()
      if ns in pf_procs and pf_procs[ns].poll() is None:
        pf_procs[ns].terminate()

    logger.info("Waiting 15s for scaler flush_summary()…")
    time.sleep(15)

    output_dir = Path(args.output_dir)
    for spec in ready_specs:
      ns = spec.namespace_name
      dst = output_dir / ns
      dst.mkdir(parents=True, exist_ok=True)
      try:
        pod_name = subprocess.check_output(f"kubectl get pods -n {ns} -l app=scaler -o jsonpath='{{.items[0].metadata.name}}'", shell=True, text=True).strip()
        if pod_name:
          try:
            with open(dst / "scaler.log", "w") as f:
              subprocess.call(f"kubectl logs -n {ns} {pod_name}", shell=True, stdout=f, stderr=subprocess.STDOUT)
          except Exception as e:
            logger.error("[%s] Failed to dump scaler logs: %s", ns, e)

          for fname in ["scaling_events.csv", "warmup_events.csv", "summary.json"]:
            try:
              subprocess.check_call(f"kubectl cp {ns}/{pod_name}:/results/{fname} {dst}/{fname}", shell=True, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
              pass
          logger.info("[%s] Results and logs copied to %s", ns, dst)
        else:
          logger.error("[%s] Scaler pod not found, cannot copy results", ns)
      except Exception as exc:
        logger.error("[%s] Failed to copy results: %s", ns, exc)

  except KeyboardInterrupt:
    logger.warning("KeyboardInterrupt received! Initiating graceful teardown...")
    raise
  except Exception as exc:
    logger.error("Unexpected error in run_batch: %s", exc)
    raise
  finally:
    for ns, proc in locust_procs.items():
      if proc.poll() is None:
        proc.terminate()
    for ns, proc in injector_procs.items():
      if proc.poll() is None:
        proc.terminate()
    
    for spec in applied:
      _kubectl_delete_namespace(spec.namespace_name)

  for spec in ready_specs:
    ns = spec.namespace_name
    port_pool.append(port_map[ns])
    
    csv_path = output_dir / ns / "scaling_events.csv"
    warmup_path = output_dir / ns / "warmup_events.csv"
    valid = False
    
    for p in [csv_path, warmup_path]:
      if p.exists():
        try:
          df = pd.read_csv(p)
          if not df.empty:
            valid = True
        except Exception:
          pass
          
    if valid:
      completed.append(ns)
    else:
      logger.warning("[%s] Empty or missing CSVs — marking failed", ns)
      failed.append(ns)

  return completed, failed


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description="Orchestrate Experiment 2 batches on K8s.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--exp1-results-dir", default="results/",
            help="Path to Exp. 1 results dir (mounted as /exp1-results in scaler)")
  parser.add_argument("--trace-dir", default="k8s/load/traces/",
            help="Directory containing *_trace.json files")
  parser.add_argument("--output-dir", default="results/exp2/",
            help="Where to write per-namespace result directories")
  parser.add_argument("--batch-size", type=int, default=5,
            help="Number of namespaces to run in parallel")
  parser.add_argument("--duration", default="30d", type=_parse_duration,
            help="Experiment duration per namespace e.g. 30d or 3600s")
  parser.add_argument("--trace-step-seconds", type=int, default=30,
            help="Seconds per trace point (TRACE_STEP_SECONDS)")
  parser.add_argument("--cpu-trace-step-seconds", type=int, default=None,
            help="Optional CPU trace step override")
  parser.add_argument("--ram-trace-step-seconds", type=int, default=None,
            help="Optional RAM trace step override")
  parser.add_argument("--models", nargs="+", default=ALL_MODELS,
            choices=ALL_MODELS,
            help="Models to run")
  parser.add_argument("--ensemble-models", nargs="+", default=None,
            choices=[m for m in ALL_MODELS if m != "Ensemble"],
            help="Strict list of sub-models for Ensemble. Overrides default behavior.")
  parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
            help="Dataset slugs to run")
  parser.add_argument("--scaling-modes", nargs="+", default=ALL_MODES,
            choices=list(MODE_SUFFIX.keys()),
            help="Scaling modes to run")
  parser.add_argument("--node-ip", default="auto",
            help="K8s node IP for NodePort access (auto will detect via kubectl)")
  parser.add_argument("--dry-run", action="store_true",
            help="Render manifests and print without applying")
  parser.add_argument("--resume", action="store_true",
            help="Resume the run from existing output_dir state.")
  parser.add_argument("--skip-warmup", action="store_true",
            help="Skip the 7-day warmup phase and start directly in evaluation mode.")
  parser.add_argument("--state-file", type=str, default=None,
            help="Path to state JSON file (defaults to output_dir/exp2_run_state.json)")
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)

  if not args.resume and not args.dry_run:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = str(Path(args.output_dir) / timestamp)

  if args.state_file is None:
    args.state_file = str(Path(args.output_dir) / "exp2_run_state.json")

  if getattr(args, "cpu_trace_step_seconds", None) is None:
    args.cpu_trace_step_seconds = args.trace_step_seconds
  if getattr(args, "ram_trace_step_seconds", None) is None:
    args.ram_trace_step_seconds = args.trace_step_seconds

  state_file = Path(args.state_file)

  if args.resume:
    if not state_file.exists():
      logger.error("Resume requested but %s not found.", state_file)
      return
    state = _load_state(state_file)
    logger.info(
      "Resuming run from %s: %d completed, %d failed, %d pending",
      state_file, len(state["completed"]), len(state["failed"]), len(state["pending"])
    )
    cfg = state["config"]
    all_specs = _build_pending(
      cfg["models"], cfg["datasets"], cfg["scaling_modes"], cfg.get("ensemble_models")
    )
    spec_by_ns = {s.namespace_name: s for s in all_specs}
    pending_specs = [spec_by_ns[ns] for ns in state["pending"] if ns in spec_by_ns]
  else:
    all_specs = _build_pending(args.models, args.datasets, args.scaling_modes, args.ensemble_models)
    state = _init_state(args, all_specs)
    state["config"]["ensemble_models"] = args.ensemble_models
    _save_state(state, state_file)
    pending_specs = all_specs
    logger.info("Starting fresh run: %d total experiments", len(pending_specs))

  if not pending_specs:
    logger.info("No pending experiments — nothing to do.")
    return

  if args.node_ip == "auto":
    try:
      node_ip = subprocess.check_output(
        "kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type==\"InternalIP\")].address}'",
        shell=True, text=True
      ).strip()
      if node_ip:
        args.node_ip = node_ip
        logger.info("Auto-detected Node IP: %s", args.node_ip)
      else:
        args.node_ip = "127.0.0.1"
        logger.info("Node IP auto-detect failed, falling back to 127.0.0.1")
    except Exception as e:
      args.node_ip = "127.0.0.1"
      logger.info("Node IP auto-detect error, falling back to 127.0.0.1: %s", e)

  port_pool: deque = deque(range(PORT_POOL_MIN, PORT_POOL_MAX + 1))

  while pending_specs:
    batch = pending_specs[: args.batch_size]
    pending_specs = pending_specs[args.batch_size :]

    batch_names = [s.namespace_name for s in batch]
    logger.info(
      "Batch: %d namespaces — %s",
      len(batch), ", ".join(batch_names),
    )

    completed_ns, failed_ns = run_batch(batch, args, state, port_pool, args.dry_run)

    state["completed"].extend(completed_ns)
    state["failed"].extend(failed_ns)
    state["pending"] = [
      ns for ns in state["pending"]
      if ns not in completed_ns and ns not in failed_ns
    ]
    _save_state(state, state_file)

    logger.info(
      "Batch done: %d completed, %d failed. "
      "Total: %d completed, %d failed, %d pending.",
      len(completed_ns), len(failed_ns),
      len(state["completed"]), len(state["failed"]), len(state["pending"]),
    )

    if pending_specs:
      logger.info("Waiting 30s before next batch…")
      time.sleep(30)

  logger.info(
    "All batches complete. Completed=%d Failed=%d",
    len(state["completed"]), len(state["failed"]),
  )
  if state["failed"]:
    logger.warning("Failed namespaces: %s", state["failed"])


if __name__ == "__main__":
  main()
