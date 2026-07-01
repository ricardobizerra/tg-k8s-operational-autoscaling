import json
import os
import threading
import time

from locust import HttpUser, task, events

_TRACE_FILE = os.environ.get("TRACE_FILE", "")
_TRACE_STEP_SECONDS = int(os.environ.get("TRACE_STEP_SECONDS", 30))
_TRACE_START_INDEX = int(os.environ.get("TRACE_START_INDEX", 0))

_rps_lock = threading.Lock()

if _TRACE_FILE:
  with open(_TRACE_FILE) as _f:
    _trace: list = json.load(_f)

  if _TRACE_START_INDEX >= len(_trace):
    raise RuntimeError(
      f"TRACE_START_INDEX={_TRACE_START_INDEX} is beyond trace length {len(_trace)}"
    )
  current_rps: int = _trace[_TRACE_START_INDEX]["rps"]
else:
  _trace = []
  current_rps = 0  # 0 indicates calibration mode (no throttling)

# Guard: stepper thread must be started exactly once
_stepper_started = False
_stepper_lock = threading.Lock()


def _rps_stepper() -> None:
  global current_rps
  if not _trace:
    return
    
  index = _TRACE_START_INDEX

  print(
    f"[locustfile] Stepper started at index={index}, "
    f"rps={current_rps}, step={_TRACE_STEP_SECONDS}s, "
    f"total_points={len(_trace)}"
  )

  while True:
    time.sleep(_TRACE_STEP_SECONDS)
    index += 1
    if index >= len(_trace):
      print("[locustfile] Trace exhausted — holding last RPS")
      break
    with _rps_lock:
      current_rps = _trace[index]["rps"]
    print(
      f"[locustfile] Step {index}/{len(_trace) - 1}: "
      f"rps={current_rps}  ({_trace[index]['timestamp']})"
    )


def _ensure_stepper_started() -> None:
  global _stepper_started
  if not _trace:
    return
    
  with _stepper_lock:
    if not _stepper_started:
      t = threading.Thread(target=_rps_stepper, daemon=True, name="rps-stepper")
      t.start()
      _stepper_started = True


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
  _ensure_stepper_started()


class WorkUser(HttpUser):
  @property
  def _user_count(self) -> int:
    try:
      return max(1, self.environment.runner.user_count)
    except Exception:
      return 1

  def wait_time(self):  # type: ignore[override]
    with _rps_lock:
      rps = current_rps
      
    if rps == 0:
      return 0
      
    per_user_rate = rps / self._user_count
    
    # Clamp to avoid zero or negative wait
    return 1.0 / max(per_user_rate, 0.01)

  @task
  def hit_work(self):
    self.client.get("/work", name="/work")
