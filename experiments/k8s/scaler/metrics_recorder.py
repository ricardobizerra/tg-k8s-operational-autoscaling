import csv
import dataclasses
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import List

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import ScalerConfig
from k8s_scaler import ScalingEvent

logger = logging.getLogger(__name__)

SCALING_EVENT_FIELDS = [f.name for f in dataclasses.fields(ScalingEvent)]


class MetricsRecorder:
  def __init__(self, config: ScalerConfig) -> None:
    self._cfg = config
    self._events: List[ScalingEvent] = []
    self._start_time = time.monotonic()

    os.makedirs(config.metrics_output_dir, exist_ok=True)

    def _init_csv(filename: str):
      path = os.path.join(config.metrics_output_dir, filename)
      file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
      f = open(path, "a", newline="")
      w = csv.DictWriter(f, fieldnames=SCALING_EVENT_FIELDS)
      if not file_exists:
        w.writeheader()
      return f, w

    self._scaling_file, self._scaling_writer = _init_csv("scaling_events.csv")
    self._warmup_file, self._warmup_writer = _init_csv("warmup_events.csv")

    logger.info("MetricsRecorder: initialized scaling and warmup logs in %s", config.metrics_output_dir)

  def record_event(self, event: ScalingEvent) -> None:
    row = dataclasses.asdict(event)
    row["timestamp"] = event.timestamp.isoformat()
    
    if event.phase == "warmup":
      self._warmup_writer.writerow(row)
      self._warmup_file.flush()
    else:
      self._events.append(event)
      self._scaling_writer.writerow(row)
      self._scaling_file.flush()

  def flush_summary(self) -> None:
    try:
      if self._events:
        csv_path = os.path.join(self._cfg.metrics_output_dir, "scaling_events.csv")
        with open(csv_path, "w", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=SCALING_EVENT_FIELDS)
          writer.writeheader()
          for e in self._events:
            r = dataclasses.asdict(e)
            r["timestamp"] = e.timestamp.isoformat()
            writer.writerow(r)
    except Exception as exc:
      logger.error("Failed to rewrite scaling_events.csv: %s", exc)

    try:
      summary = self._compute_summary()
      path = os.path.join(self._cfg.metrics_output_dir, "summary.json")
      with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
      logger.info("Summary written to %s", path)
    except Exception as exc:
      logger.error("Failed to write summary: %s", exc)
    finally:
      try:
        self._scaling_file.flush()
        self._scaling_file.close()
        self._warmup_file.flush()
        self._warmup_file.close()
      except Exception:
        pass

  def _compute_summary(self) -> dict:
    events = self._events
    cfg = self._cfg
    total = len(events)
    duration = time.monotonic() - self._start_time

    cpu_pred = [e.predicted_cpu_cores for e in events]
    cpu_act  = [e.actual_cpu_cores for e in events]
    ram_pred = [e.predicted_ram_gb for e in events]
    ram_act  = [e.actual_ram_gb for e in events]

    def _pred_metrics(pred, actual):
      if len(pred) < 2:
        return {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "BIAS": 0.0}
      try:
        mae  = float(mean_absolute_error(actual, pred))
        rmse = float(mean_squared_error(actual, pred, squared=False))
        r2   = float(r2_score(actual, pred))
        bias = float(sum(p - a for p, a in zip(pred, actual)) / len(pred))
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "BIAS": bias}
      except Exception:
        return {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "BIAS": 0.0}

    sla_cpu_count = sum(1 for e in events if e.sla_cpu_violation)
    sla_ram_count = sum(1 for e in events if e.sla_ram_violation)

    step_min = cfg.scrape_interval_seconds / 60.0
    overprov_cpu = sum(
      step_min for e in events
      if e.current_replicas > math.ceil(
        max(e.actual_cpu_cores, 1e-9) / cfg.cpu_target_per_replica
      )
    )
    overprov_ram = sum(
      step_min for e in events
      if e.current_memory_mb > math.ceil(e.actual_ram_gb * 1024)
    )

    scale_up_h   = sum(1 for e in events if e.action_horizontal == "scale_up")
    scale_down_h = sum(1 for e in events if e.action_horizontal == "scale_down")
    scale_up_v   = sum(1 for e in events if e.action_vertical == "scale_up")
    scale_down_v = sum(1 for e in events if e.action_vertical == "scale_down")

    osc_h = self._count_oscillations(
      [(e.timestamp, e.action_horizontal) for e in events
       if e.action_horizontal in ("scale_up", "scale_down")],
      cfg.scale_down_cooldown_seconds,
    )
    osc_v = self._count_oscillations(
      [(e.timestamp, e.action_vertical) for e in events
       if e.action_vertical in ("scale_up", "scale_down")],
      cfg.vertical_cooldown_seconds,
    )
    
    pod_restarts = sum(e.pod_restart_count for e in events)
    total_eval_minutes = total * step_min
    waste_ratio_cpu = overprov_cpu / total_eval_minutes if total_eval_minutes else 0.0
    waste_ratio_ram = overprov_ram / total_eval_minutes if total_eval_minutes else 0.0
    
    total_eval_seconds = duration
    mtbv_cpu = total_eval_seconds / sla_cpu_count if sla_cpu_count else float('inf')
    mtbv_ram = total_eval_seconds / sla_ram_count if sla_ram_count else float('inf')

    avg_replicas = (
      sum(e.current_replicas for e in events) / total if total else 0.0
    )
    avg_memory_mb = (
      sum(e.current_memory_mb for e in events) / total if total else 0.0
    )
    refit_skip = max((e.refit_skipped for e in events), default=0)

    fit_time_cpu = [e.fit_time_cpu_seconds for e in events if e.fit_time_cpu_seconds > 0]
    predict_time_cpu = [e.predict_time_cpu_seconds for e in events if e.predict_time_cpu_seconds > 0]
    fit_time_ram = [e.fit_time_ram_seconds for e in events if e.fit_time_ram_seconds > 0]
    predict_time_ram = [e.predict_time_ram_seconds for e in events if e.predict_time_ram_seconds > 0]

    return {
      "namespace": cfg.namespace,
      "model_name": cfg.model_name,
      "dataset_slug": cfg.dataset_slug,
      "scaling_mode": cfg.scaling_mode,
      "total_cycles": total,
      "total_duration_seconds": round(duration, 1),
      "prediction_metrics": {
        "cpu": _pred_metrics(cpu_pred, cpu_act),
        "ram": _pred_metrics(ram_pred, ram_act),
      },
      "operational_metrics": {
        "sla_cpu_violation_count": sla_cpu_count,
        "sla_cpu_violation_rate": sla_cpu_count / total if total else 0.0,
        "mtbv_cpu_seconds": round(mtbv_cpu, 1) if mtbv_cpu != float('inf') else None,
        "sla_ram_violation_count": sla_ram_count,
        "sla_ram_violation_rate": sla_ram_count / total if total else 0.0,
        "mtbv_ram_seconds": round(mtbv_ram, 1) if mtbv_ram != float('inf') else None,
        "overprovision_cpu_minutes": round(overprov_cpu, 2),
        "waste_ratio_cpu": round(waste_ratio_cpu, 4),
        "overprovision_ram_minutes": round(overprov_ram, 2),
        "waste_ratio_ram": round(waste_ratio_ram, 4),
        "oscillation_horizontal_count": osc_h,
        "oscillation_vertical_count": osc_v,
        "scale_up_horizontal_count": scale_up_h,
        "scale_down_horizontal_count": scale_down_h,
        "scale_up_vertical_count": scale_up_v,
        "scale_down_vertical_count": scale_down_v,
        "pod_restart_count": pod_restarts,
        "avg_replicas": round(avg_replicas, 2),
        "avg_memory_mb": round(avg_memory_mb, 2),
        "refit_skip_count": refit_skip,
        "avg_fit_time_cpu_seconds": sum(fit_time_cpu) / len(fit_time_cpu) if fit_time_cpu else 0.0,
        "max_fit_time_cpu_seconds": max(fit_time_cpu) if fit_time_cpu else 0.0,
        "avg_predict_time_cpu_seconds": sum(predict_time_cpu) / len(predict_time_cpu) if predict_time_cpu else 0.0,
        "max_predict_time_cpu_seconds": max(predict_time_cpu) if predict_time_cpu else 0.0,
        "avg_fit_time_ram_seconds": sum(fit_time_ram) / len(fit_time_ram) if fit_time_ram else 0.0,
        "max_fit_time_ram_seconds": max(fit_time_ram) if fit_time_ram else 0.0,
        "avg_predict_time_ram_seconds": sum(predict_time_ram) / len(predict_time_ram) if predict_time_ram else 0.0,
        "max_predict_time_ram_seconds": max(predict_time_ram) if predict_time_ram else 0.0,
      },
    }

  @staticmethod
  def _count_oscillations(actions: list, cooldown_seconds: float) -> int:
    oscillations = 0
    for i in range(1, len(actions)):
      t_prev, a_prev = actions[i - 1]
      t_curr, a_curr = actions[i]
      delta = (t_curr - t_prev).total_seconds()
      if a_prev != a_curr and delta <= cooldown_seconds:
        oscillations += 1
    return oscillations
