import logging
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone

import pandas as pd

import constants
from config import load_config
from predictor import Predictor
from prometheus_client import PrometheusClient
from k8s_scaler import KubernetesScaler
from metrics_recorder import MetricsRecorder

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
  datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger("scaler.main")


def main() -> None:
  cfg = load_config()
  logger.info(
    "Scaler starting: namespace=%s model=%s dataset=%s mode=%s",
    cfg.namespace,
    cfg.model_name,
    cfg.dataset_slug,
    cfg.scaling_mode,
  )
  
  vm_cpu_cap = cfg.max_replicas * cfg.cpu_target_per_replica
  vm_ram_cap = cfg.max_memory_mb / 1024.0

  app_letter = "A"
  if "b-" in cfg.dataset_slug.lower():
    app_letter = "B"
  elif "c-" in cfg.dataset_slug.lower():
    app_letter = "C"

  real_cpu_cap = float(constants.APP_TOTAL_CPU_CORES.get(app_letter, vm_cpu_cap))
  real_ram_cap = float(constants.APP_TOTAL_RAM_SIZE_GB.get(app_letter, vm_ram_cap))

  logger.info(
    "Scale Engine active: VM Cores=%.2f, VM RAM=%.2fGB | Real Cores=%.2f, Real RAM=%.2fGB",
    vm_cpu_cap,
    vm_ram_cap,
    real_cpu_cap,
    real_ram_cap,
  )

  prometheus = PrometheusClient(cfg.prometheus_url, config=cfg)
  predictor = Predictor(cfg)
  scaler = KubernetesScaler(cfg)
  recorder = MetricsRecorder(cfg)

  def _sigterm_handler(signum, frame):
    logger.info("SIGTERM received — flushing summary and exiting")
    signal.alarm(10)
    try:
      recorder.flush_summary()
    except Exception as exc:
      logger.error("flush_summary failed: %s", exc)
    sys.exit(0)

  signal.signal(signal.SIGTERM, _sigterm_handler)

  logger.info(
    "Bootstrapping history: %d minutes at %ds steps",
    cfg.bootstrap_minutes,
    cfg.scrape_interval_seconds,
  )
  cpu_bootstrap = prometheus.bootstrap_history(
    cfg.namespace, "cpu", cfg.bootstrap_minutes, cfg.scrape_interval_seconds
  )
  ram_bootstrap = prometheus.bootstrap_history(
    cfg.namespace, "ram", cfg.bootstrap_minutes, cfg.scrape_interval_seconds
  )
  logger.info(
    "Bootstrap complete: cpu=%d rows, ram=%d rows",
    len(cpu_bootstrap),
    len(ram_bootstrap),
  )
  
  if not cpu_bootstrap.empty:
    cpu_bootstrap["Total_Usage"] = cpu_bootstrap["Total_Usage"] * (real_cpu_cap / vm_cpu_cap)
  if not ram_bootstrap.empty:
    ram_bootstrap["Total_Usage"] = ram_bootstrap["Total_Usage"] * (real_ram_cap / vm_ram_cap)

  cpu_deque: deque = deque(
    cpu_bootstrap.to_dict("records"), maxlen=cfg.max_history_points
  )
  ram_deque: deque = deque(
    ram_bootstrap.to_dict("records"), maxlen=cfg.max_history_points
  )

  cycle = 0
  while True:
    loop_start = time.monotonic()

    actual_cpu = prometheus.query_cpu_cores(cfg.namespace)
    actual_ram = prometheus.query_ram_gb(cfg.namespace)

    if scaler.detect_oom_kills():
      logger.error("Crash detected! Injecting synthetic memory spike to recover autoscaler.")
      actual_ram = cfg.max_memory_mb / 1024.0

    actual_real_cpu = actual_cpu * (real_cpu_cap / vm_cpu_cap)
    actual_real_ram = actual_ram * (real_ram_cap / vm_ram_cap)

    now_utc = datetime.now(tz=timezone.utc)
    cpu_deque.append({"timestamp_date_format": now_utc, "Total_Usage": actual_real_cpu})
    ram_deque.append({"timestamp_date_format": now_utc, "Total_Usage": actual_real_ram})

    cpu_df = pd.DataFrame(list(cpu_deque))
    ram_df = pd.DataFrame(list(ram_deque))

    if cfg.scaling_mode == "combined":
      if cfg.dataset_slug.endswith("-cpu"):
        predicted_cpu = predictor.fit_predict(cpu_df)
        predicted_ram = predictor.fit_predict_secondary(ram_df)
      else:
        predicted_ram = predictor.fit_predict(ram_df)
        predicted_cpu = predictor.fit_predict_secondary(cpu_df)
    elif cfg.dataset_slug.endswith("-cpu"):
      predicted_cpu = predictor.fit_predict(cpu_df)
      predicted_ram = actual_ram 
    else:
      predicted_ram = predictor.fit_predict(ram_df)
      predicted_cpu = actual_real_cpu

    predicted_vm_cpu = predicted_cpu * (vm_cpu_cap / real_cpu_cap)
    predicted_vm_ram = predicted_ram * (vm_ram_cap / real_ram_cap)

    event = scaler.reconcile(
      predicted_vm_cpu,
      actual_cpu,
      predicted_vm_ram,
      actual_ram,
      cycle_number=cycle,
    )
    event.cycle_number = cycle
    event.refit_skipped = predictor.refit_skip_count
    event.selected_model = predictor.last_selected_model
    event.phase = "warmup" if cycle < cfg.warmup_cycles else "evaluation"

    if cfg.dataset_slug.endswith("-cpu"):
      event.fit_time_cpu_seconds = predictor.last_fit_time
      event.predict_time_cpu_seconds = predictor.last_predict_time
      event.fit_time_ram_seconds = predictor.last_fit_time_secondary
      event.predict_time_ram_seconds = predictor.last_predict_time_secondary
    else:
      event.fit_time_ram_seconds = predictor.last_fit_time
      event.predict_time_ram_seconds = predictor.last_predict_time
      event.fit_time_cpu_seconds = predictor.last_fit_time_secondary
      event.predict_time_cpu_seconds = predictor.last_predict_time_secondary

    recorder.record_event(event)

    logger.info(
      "Cycle %d | cpu=%.3f→%.3f replicas=%d→%d | "
      "ram=%.3fGB→%.3fGB mem=%dMB→%dMB | "
      "h=%s v=%s | sla_cpu=%s sla_ram=%s",
      cycle,
      actual_cpu, predicted_cpu, event.current_replicas, event.desired_replicas,
      actual_ram, predicted_ram, event.current_memory_mb, event.desired_memory_mb,
      event.action_horizontal, event.action_vertical,
      event.sla_cpu_violation, event.sla_ram_violation,
    )

    cycle += 1

    elapsed = time.monotonic() - loop_start
    sleep_for = max(0.0, cfg.scrape_interval_seconds - elapsed)
    time.sleep(sleep_for)


if __name__ == "__main__":
  main()
