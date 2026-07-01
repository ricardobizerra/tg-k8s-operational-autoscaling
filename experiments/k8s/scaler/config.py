import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ScalerConfig:
  namespace: str = ""
  deployment_name: str = "target-app"
  model_name: str = ""
  ensemble_models: List[str] = field(default_factory=list)
  dataset_slug: str = ""
  scaling_mode: str = "horizontal"  # "horizontal" | "vertical" | "combined"
  exp1_results_dir: str = "/exp1-results"
  prometheus_url: str = "http://prometheus.tcc-infra.svc.cluster.local:9090"
  cpu_query_template: str = (
    'sum(rate(container_cpu_usage_seconds_total'
    '{{namespace="{namespace}",container="target-app"}}[1m]))'
  )
  ram_query_template: str = (
    'sum(container_memory_working_set_bytes'
    '{{namespace="{namespace}",container="target-app"}}) / 1073741824'
  )
  
  scrape_interval_seconds: int = 60
  refit_timeout_seconds: int = 25
  max_history_points: int = 500
  bootstrap_minutes: int = 30
  
  min_replicas: int = 1
  max_replicas: int = 10
  cpu_target_per_replica: float = 0.5
  
  min_memory_mb: int = 64
  max_memory_mb: int = 4096
  memory_step_mb: int = 64
  
  scale_up_cooldown_seconds: int = 60
  scale_down_cooldown_seconds: int = 180
  vertical_cooldown_seconds: int = 300
  
  sla_cpu_threshold: float = 0.0
  sla_ram_threshold_mb: float = 0.0
  
  metrics_output_dir: str = "/results"
  warmup_cycles: int = 0


def _env_str(key: str, default: str = "") -> str:
  return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
  raw = os.environ.get(key)
  return int(raw) if raw is not None else default


def _env_float(key: str, default: float) -> float:
  raw = os.environ.get(key)
  return float(raw) if raw is not None else default


def load_config() -> ScalerConfig:
  ensemble_raw = _env_str("ENSEMBLE_MODELS", "")
  ensemble_models = (
    [m.strip() for m in ensemble_raw.split(",") if m.strip()]
    if ensemble_raw
    else []
  )

  cpu_target = _env_float("CPU_TARGET_PER_REPLICA", 0.5)
  max_memory_mb = _env_int("MAX_MEMORY_MB", 4096)

  sla_cpu = _env_float("SLA_CPU_THRESHOLD", 0.0) or (0.9 * cpu_target)
  sla_ram = _env_float("SLA_RAM_THRESHOLD_MB", 0.0) or (0.9 * max_memory_mb)

  cfg = ScalerConfig(
    namespace=_env_str("NAMESPACE"),
    deployment_name=_env_str("DEPLOYMENT_NAME", "target-app"),
    model_name=_env_str("MODEL_NAME"),
    ensemble_models=ensemble_models,
    dataset_slug=_env_str("DATASET_SLUG"),
    scaling_mode=_env_str("SCALING_MODE", "horizontal"),
    exp1_results_dir=_env_str("EXP1_RESULTS_DIR", "/exp1-results"),
    prometheus_url=_env_str(
      "PROMETHEUS_URL",
      "http://prometheus.tcc-infra.svc.cluster.local:9090",
    ),
    cpu_query_template=_env_str(
      "CPU_QUERY_TEMPLATE",
      ScalerConfig.cpu_query_template,
    ),
    ram_query_template=_env_str(
      "RAM_QUERY_TEMPLATE",
      ScalerConfig.ram_query_template,
    ),
    scrape_interval_seconds=_env_int("SCRAPE_INTERVAL_SECONDS", 60),
    refit_timeout_seconds=_env_int("REFIT_TIMEOUT_SECONDS", 25),
    max_history_points=_env_int("MAX_HISTORY_POINTS", 500),
    bootstrap_minutes=_env_int("BOOTSTRAP_MINUTES", 30),
    min_replicas=_env_int("MIN_REPLICAS", 1),
    max_replicas=_env_int("MAX_REPLICAS", 10),
    cpu_target_per_replica=cpu_target,
    min_memory_mb=_env_int("MIN_MEMORY_MB", 64),
    max_memory_mb=max_memory_mb,
    memory_step_mb=_env_int("MEMORY_STEP_MB", 64),
    scale_up_cooldown_seconds=_env_int("SCALE_UP_COOLDOWN_SECONDS", 60),
    scale_down_cooldown_seconds=_env_int("SCALE_DOWN_COOLDOWN_SECONDS", 180),
    vertical_cooldown_seconds=_env_int("VERTICAL_COOLDOWN_SECONDS", 300),
    sla_cpu_threshold=sla_cpu,
    sla_ram_threshold_mb=sla_ram,
    metrics_output_dir=_env_str("METRICS_OUTPUT_DIR", "/results"),
    warmup_cycles=_env_int("WARMUP_CYCLES", 0),
  )

  _validate(cfg)
  return cfg


def _validate(cfg: ScalerConfig) -> None:
  if not cfg.namespace:
    raise ValueError("NAMESPACE env var is required")
  if not cfg.model_name:
    raise ValueError("MODEL_NAME env var is required")
  if not cfg.dataset_slug:
    raise ValueError("DATASET_SLUG env var is required")
  if cfg.scaling_mode not in ("horizontal", "vertical", "combined"):
    raise ValueError(
      f"SCALING_MODE must be horizontal|vertical|combined, got '{cfg.scaling_mode}'"
    )
  if cfg.model_name == "Ensemble" and not cfg.ensemble_models:
    raise ValueError(
      "ENSEMBLE_MODELS must be set when MODEL_NAME=Ensemble"
    )
