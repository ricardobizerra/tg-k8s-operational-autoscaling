import argparse
import csv
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone

MAX_RETRIES = 10
RETRY_BACKOFF_S = 3


def _post_set_memory_kubectl(namespace: str, pod_name: str, memory_mb: int) -> tuple[int, float]:
  url = f"http://localhost:8000/set-memory?mb={memory_mb}"
  t0 = time.monotonic()
  
  cmd = [
    "kubectl", "exec", "-n", namespace, pod_name, "--",
    "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-X", "POST", url
  ]
  
  result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
  latency_ms = (time.monotonic() - t0) * 1000
  
  if result.returncode != 0:
    raise RuntimeError(f"kubectl exec failed: {result.stderr}")
    
  status_code = int(result.stdout.strip() or 0)
  if status_code < 200 or status_code >= 300:
    raise RuntimeError(f"HTTP {status_code}")
    
  return status_code, latency_ms


def _inject_with_retry(namespace: str, pod_name: str, memory_mb: int) -> tuple[str, float]:
  for attempt in range(MAX_RETRIES):
    try:
      status_code, latency_ms = _post_set_memory_kubectl(namespace, pod_name, memory_mb)
      return str(status_code), latency_ms
    except Exception as exc:
      print(
        f"  Error on attempt {attempt + 1}/{MAX_RETRIES} for pod {pod_name}: {exc}, "
        f"retrying in {RETRY_BACKOFF_S}s …",
        file=sys.stderr,
      )
    time.sleep(RETRY_BACKOFF_S)

  # All retries exhausted
  return "ERROR", 0.0


def _open_log(trace_path: str) -> tuple[object, object]:
  log_dir = os.path.dirname(os.path.abspath(trace_path))
  log_path = os.path.join(log_dir, "ram_injection_log.csv")
  file_exists = os.path.isfile(log_path)
  f = open(log_path, "a", newline="")
  writer = csv.DictWriter(
    f,
    fieldnames=["timestamp", "trace_index", "memory_mb", "status_code", "latency_ms"],
  )
  if not file_exists:
    writer.writeheader()
  print(f"Logging to: {log_path}")
  return f, writer


def _get_active_pod_names(namespace: str) -> list[str]:
  names = []
  try:
    cmd = [
      "kubectl", "get", "pods", "-n", namespace,
      "-l", "app=target-app",
      "--field-selector=status.phase=Running",
      "-o", "jsonpath={.items[*].metadata.name}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode == 0 and result.stdout.strip():
      # Filter out empty names
      names = [name for name in result.stdout.strip().split() if name]
  except Exception as e:
    print(f"  [!] Failed to list pods: {e}", file=sys.stderr)
  return names

def run(trace: list, namespace: str, step_seconds: int, start_index: int, log_writer) -> None:
  total = len(trace)

  print(f"  [startup] waiting for pods in {namespace} …")
  
  for _ in range(30):
    names = _get_active_pod_names(namespace)
    if names:
      break
    time.sleep(2)
    
  for point in trace[start_index:]:
    idx = point["index"]
    memory_mb = point["memory_mb"]
    ts_wall = datetime.now(tz=timezone.utc).isoformat()

    active_names = _get_active_pod_names(namespace)
    num_pods = len(active_names)
    
    if num_pods == 0:
      print(f"[{idx}/{total - 1}] {point['timestamp']} memory_mb={memory_mb} -> NO ACTIVE PODS")
      log_writer.writerow({
        "timestamp": ts_wall,
        "trace_index": idx,
        "memory_mb": memory_mb,
        "status_code": "NO_PODS",
        "latency_ms": 0.0,
      })
      time.sleep(step_seconds)
      continue

    memory_per_pod = int(memory_mb / num_pods)

    print(
      f"[{idx}/{total - 1}] {point['timestamp']}  "
      f"total_mb={memory_mb}  pods={num_pods}"
    )

    for pod_name in active_names:
      status_code, latency = _inject_with_retry(namespace, pod_name, memory_per_pod)
      
      log_writer.writerow({
        "timestamp": ts_wall,
        "trace_index": idx,
        "memory_mb": memory_mb,
        "status_code": status_code,
        "latency_ms": round(latency, 2),
      })
      print(f"  → pod={pod_name} status={status_code}  latency={latency:.1f}ms")

    time.sleep(step_seconds)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description="Step through a RAM trace JSON and inject memory into target-app replicas.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--trace", required=True, help="Path to RAM trace JSON")
  parser.add_argument(
    "--namespace", required=True, help="Kubernetes namespace to locate target-app pods"
  )
  parser.add_argument(
    "--host", required=False, help="(Legacy) Base URL of target-app, ignored"
  )
  parser.add_argument(
    "--step-seconds",
    type=int,
    default=30,
    help="Seconds to wait between injections",
  )
  parser.add_argument(
    "--start-index",
    type=int,
    default=0,
    help="Resume from trace index N (skips first N points)",
  )
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)

  with open(args.trace) as f:
    trace = json.load(f)

  if args.start_index >= len(trace):
    print(
      f"ERROR: --start-index {args.start_index} is beyond trace length {len(trace)}",
      file=sys.stderr,
    )
    sys.exit(1)

  print(
    f"RAM injector started: {len(trace)} points, "
    f"starting at index {args.start_index}, "
    f"step={args.step_seconds}s, namespace={args.namespace}"
  )

  log_file, log_writer = _open_log(args.trace)
  try:
    run(trace, args.namespace, args.step_seconds, args.start_index, log_writer)
  finally:
    log_file.flush()
    log_file.close()

  print("RAM injector finished.")


if __name__ == "__main__":
  main()
