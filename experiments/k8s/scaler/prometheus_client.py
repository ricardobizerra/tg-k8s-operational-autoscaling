import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from config import ScalerConfig

logger = logging.getLogger(__name__)


class PrometheusClient:
  def __init__(self, url: str, config: Optional[ScalerConfig] = None) -> None:
    self._url = url.rstrip("/")
    self._cfg = config

  def query_instant(self, promql: str) -> float:
    try:
      resp = requests.get(
        f"{self._url}/api/v1/query",
        params={"query": promql},
        timeout=10,
      )
      resp.raise_for_status()
      data = resp.json()
      result = data.get("data", {}).get("result", [])
      if not result:
        logger.warning("Empty Prometheus result for query: %s", promql)
        return 0.0
      return float(result[0]["value"][1])
    except Exception as exc:
      logger.error("Prometheus instant query failed: %s — %s", promql, exc)
      return 0.0

  def query_range(
    self,
    promql: str,
    start: datetime,
    end: datetime,
    step_seconds: int,
  ) -> pd.DataFrame:
    try:
      resp = requests.get(
        f"{self._url}/api/v1/query_range",
        params={
          "query": promql,
          "start": start.timestamp(),
          "end": end.timestamp(),
          "step": str(step_seconds),
        },
        timeout=30,
      )
      resp.raise_for_status()
      data = resp.json()
      result = data.get("data", {}).get("result", [])
      if not result:
        logger.warning("Empty Prometheus range result for query: %s", promql)
        return pd.DataFrame(columns=["timestamp", "value"])
      rows = [
        {"timestamp": datetime.fromtimestamp(ts, tz=timezone.utc), "value": float(val)}
        for ts, val in result[0]["values"]
      ]
      return pd.DataFrame(rows)
    except Exception as exc:
      logger.error("Prometheus range query failed: %s — %s", promql, exc)
      return pd.DataFrame(columns=["timestamp", "value"])

  def query_cpu_cores(self, namespace: str) -> float:
    if self._cfg is not None:
      promql = self._cfg.cpu_query_template.format(namespace=namespace)
    else:
      promql = (
        f'sum(rate(container_cpu_usage_seconds_total'
        f'{{namespace="{namespace}",container="target-app"}}[1m]))'
      )
    return self.query_instant(promql)

  def query_ram_gb(self, namespace: str) -> float:
    if self._cfg is not None:
      promql = self._cfg.ram_query_template.format(namespace=namespace)
    else:
      promql = (
        f'sum(container_memory_working_set_bytes'
        f'{{namespace="{namespace}",container="target-app"}}) / 1073741824'
      )
    return self.query_instant(promql)

  def bootstrap_history(
    self,
    namespace: str,
    metric: str,
    duration_minutes: int,
    step_seconds: int,
  ) -> pd.DataFrame:
    end = datetime.now(tz=timezone.utc)
    start = datetime(
      end.year, end.month, end.day, end.hour, end.minute, end.second,
      tzinfo=timezone.utc,
    )
    import time as _time
    start_ts = end.timestamp() - duration_minutes * 60
    start = datetime.fromtimestamp(start_ts, tz=timezone.utc)

    if metric == "cpu":
      raw_df = self.query_range(
        self._cfg.cpu_query_template.format(namespace=namespace)
        if self._cfg
        else (
          f'sum(rate(container_cpu_usage_seconds_total'
          f'{{namespace="{namespace}",container="target-app"}}[1m]))'
        ),
        start,
        end,
        step_seconds,
      )
    elif metric == "ram":
      raw_df = self.query_range(
        self._cfg.ram_query_template.format(namespace=namespace)
        if self._cfg
        else (
          f'sum(container_memory_working_set_bytes'
          f'{{namespace="{namespace}",container="target-app"}}) / 1073741824'
        ),
        start,
        end,
        step_seconds,
      )
    else:
      raise ValueError(f"Unknown metric '{metric}'; expected 'cpu' or 'ram'")

    if raw_df.empty:
      return pd.DataFrame(columns=["timestamp_date_format", "Total_Usage"])

    history = raw_df.rename(
      columns={"timestamp": "timestamp_date_format", "value": "Total_Usage"}
    )
    return history.reset_index(drop=True)
