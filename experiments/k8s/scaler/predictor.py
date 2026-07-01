import logging
import signal
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config import ScalerConfig

from training.pipeline import instantiate_model
from training import phase_io

logger = logging.getLogger(__name__)

SLUG_TO_DATASET_NAME = {
  "a-cpu": "cegedim_application_A_CPU_exp1",
  "a-ram": "cegedim_application_A_RAM_exp1",
  "b-cpu": "cegedim_application_B_CPU_exp1",
  "b-ram": "cegedim_application_B_RAM_exp1",
  "c-cpu": "cegedim_application_C_CPU_exp1",
  "c-ram": "cegedim_application_C_RAM_exp1",
}

DATASET_CONFIG = {
  "a-cpu": {"freq": "1H",   "unit": "cores"},
  "a-ram": {"freq": "40T",  "unit": "gb"},
  "b-cpu": {"freq": "1H",   "unit": "cores"},
  "b-ram": {"freq": "40T",  "unit": "gb"},
  "c-cpu": {"freq": "1H",   "unit": "cores"},
  "c-ram": {"freq": "40T",  "unit": "gb"},
}

ENSEMBLE_BACKWARD_WINDOW = {"cpu": 24, "ram": 36}
ENSEMBLE_FORWARD_WINDOW = 1

TARGET_COL = "Total_Usage"
MIN_HISTORY_POINTS = 2


def _resolve_run_dir(exp1_results_dir: str, dataset_slug: str, model_save_name: str) -> Path:
  dataset_name = SLUG_TO_DATASET_NAME.get(dataset_slug)
  if dataset_name is None:
    raise ValueError(f"Unknown dataset_slug '{dataset_slug}'")

  try:
    run_dir = phase_io.get_latest_run_dir(exp1_results_dir, dataset_name, model_save_name)
    return Path(run_dir)
  except FileNotFoundError:
    if model_save_name.startswith("Ensemble[") and model_save_name.endswith("]"):
      import itertools
      components_str = model_save_name[len("Ensemble["):-1]
      components = components_str.split("+")
      for perm in itertools.permutations(components):
        perm_save_name = f"Ensemble[{'+'.join(perm)}]"
        try:
          run_dir = phase_io.get_latest_run_dir(exp1_results_dir, dataset_name, perm_save_name)
          return Path(run_dir)
        except FileNotFoundError:
          pass
          
    raise FileNotFoundError(
      f"No Exp. 1 run found for model='{model_save_name}' "
      f"(or its permutations) in dataset='{dataset_name}' in '{exp1_results_dir}'. "
      f"Run the Exp. 1 pipeline (stages: tuning) for this combination first."
    )


def _model_save_name(model_name: str, ensemble_models: list) -> str:
  if model_name == "Ensemble" and ensemble_models:
    components_str = "+".join(ensemble_models)
    return f"Ensemble[{components_str}]"
  return model_name


def _other_slug(dataset_slug: str) -> str:
  app, metric = dataset_slug.split("-", 1)
  other = "ram" if metric == "cpu" else "cpu"
  return f"{app}-{other}"


def _build_model(config: ScalerConfig, dataset_slug: str):
  save_name = _model_save_name(config.model_name, config.ensemble_models)
  run_dir = _resolve_run_dir(config.exp1_results_dir, dataset_slug, save_name)
  best_params = phase_io.load_best_params(str(run_dir))

  if config.model_name == "Ensemble":
    metric_type = dataset_slug.split("-")[1]  # "cpu" or "ram"
    bw = best_params.get("backward_window", ENSEMBLE_BACKWARD_WINDOW[metric_type])
    inner_instances = []
    for name in config.ensemble_models:
      inner_run_dir = _resolve_run_dir(config.exp1_results_dir, dataset_slug, name)
      inner_params = phase_io.load_best_params(str(inner_run_dir))
      inner_instances.append(instantiate_model(name, inner_params))
    model = instantiate_model(
      "Ensemble",
      {"backward_window": bw, "forward_window": ENSEMBLE_FORWARD_WINDOW, "online_mode": True},
      ensemble_models_instances=inner_instances,
    )
  else:
    model = instantiate_model(config.model_name, best_params)

  logger.info(
    "Model loaded: %s | dataset_slug=%s | run_dir=%s",
    save_name,
    dataset_slug,
    run_dir,
  )
  return model


class _TimeoutError(Exception):
  pass


def _fit_with_timeout(model, history: pd.DataFrame, freq: str, timeout_seconds: int) -> bool:
  def _handler(signum, frame):
    raise _TimeoutError()

  signal.signal(signal.SIGALRM, _handler)
  signal.alarm(timeout_seconds)
  try:
    model.fit(history, TARGET_COL, freq)
    signal.alarm(0)
    return True
  except _TimeoutError:
    signal.alarm(0)
    logger.warning(
      "Re-fit timeout (%ds) exceeded — using previous prediction", timeout_seconds
    )
    return False
  except Exception as exc:
    signal.alarm(0)
    logger.error("Re-fit raised an exception: %s", exc)
    return False


def _make_next_timestamp_df(history: pd.DataFrame, freq: str) -> pd.DataFrame:
  last_ts = pd.to_datetime(history["timestamp_date_format"].iloc[-1])
  offset = pd.tseries.frequencies.to_offset(freq)
  next_ts = last_ts + offset
  return pd.DataFrame({"timestamp_date_format": [next_ts], TARGET_COL: [0.0]})


def _last_observed_value(history: pd.DataFrame) -> float:
  try:
    return max(0.0, float(history[TARGET_COL].iloc[-1]))
  except Exception:
    return 0.0


def _normalize_history(history: pd.DataFrame, freq: str) -> pd.DataFrame:
  if history.empty:
    return history

  normalized = history.copy()
  normalized["timestamp_date_format"] = pd.to_datetime(
    normalized["timestamp_date_format"], utc=True
  ).dt.tz_convert(None)
  normalized = normalized.sort_values("timestamp_date_format")
  normalized["timestamp_date_format"] = normalized["timestamp_date_format"].dt.floor(freq)
  normalized = (
    normalized.groupby("timestamp_date_format", as_index=False)[TARGET_COL]
    .mean()
    .sort_values("timestamp_date_format")
  )

  start = normalized["timestamp_date_format"].iloc[0]
  end = normalized["timestamp_date_format"].iloc[-1]
  full_index = pd.date_range(start=start, end=end, freq=freq)
  normalized = (
    normalized.set_index("timestamp_date_format")
    .reindex(full_index)
    .interpolate(limit_direction="both")
    .reset_index()
    .rename(columns={"index": "timestamp_date_format"})
  )
  return normalized


def _log_history_sample(label: str, history: pd.DataFrame, freq: str) -> None:
  if history.empty:
    logger.info("%s: empty history (freq=%s)", label, freq)
    return

  head = history.head(3).to_dict("records")
  tail = history.tail(3).to_dict("records")
  logger.info(
    "%s: rows=%d freq=%s head=%s tail=%s",
    label,
    len(history),
    freq,
    head,
    tail,
  )


class Predictor:
  def __init__(self, config: ScalerConfig) -> None:
    self._config = config
    self._freq = DATASET_CONFIG[config.dataset_slug]["freq"]
    self._live_freq = f"{self._config.scrape_interval_seconds}S"
    self._model = _build_model(config, config.dataset_slug)

    self._model_secondary = None
    self._freq_secondary: Optional[str] = None
    if config.scaling_mode == "combined":
      other_slug = _other_slug(config.dataset_slug)
      self._model_secondary = _build_model(config, other_slug)
      self._freq_secondary = DATASET_CONFIG[other_slug]["freq"]
      logger.info("Secondary model loaded for combined mode: slug=%s", other_slug)

    self._last_prediction: float = 0.0
    self._last_prediction_secondary: float = 0.0
    self.last_fit_time = 0.0
    self.last_predict_time = 0.0
    self.last_fit_time_secondary = 0.0
    self.last_predict_time_secondary = 0.0
    self._cycle_count: int = 0
    self._refit_skip_count: int = 0
    self._last_selected_model: str = config.model_name

  def fit_predict(self, history: pd.DataFrame) -> float:
    history = _normalize_history(history, self._live_freq)
    _log_history_sample("fit_predict.history", history, self._live_freq)

    if len(history) < MIN_HISTORY_POINTS:
      logger.warning(
        "History too short (%d rows) to fit; returning last prediction",
        len(history),
      )
      fallback = _last_observed_value(history)
      self._last_prediction = max(self._last_prediction, fallback)
      return self._last_prediction

    t0_fit = time.monotonic()
    ok = _fit_with_timeout(
      self._model, history, self._live_freq, self._config.refit_timeout_seconds
    )
    self.last_fit_time = time.monotonic() - t0_fit
    
    if not ok:
      self._refit_skip_count += 1
      self._last_prediction = max(self._last_prediction, _last_observed_value(history))
      return self._last_prediction

    test_df = _make_next_timestamp_df(history, self._live_freq)
    _log_history_sample("fit_predict.test_df", test_df, self._live_freq)
    try:
      t0_pred = time.monotonic()
      pred = float(self._model.predict(test_df, TARGET_COL, self._live_freq).iloc[0])
      self.last_predict_time = time.monotonic() - t0_pred
    except Exception as exc:
      logger.error("predict() failed: %s", exc)
      self._refit_skip_count += 1
      self._last_prediction = max(self._last_prediction, _last_observed_value(history))
      return self._last_prediction

    logger.info("fit_predict.raw_pred=%s freq=%s", pred, self._live_freq)
    self._last_prediction = max(0.0, pred)

    if hasattr(self._model, "last_selected_model_name"):
      self._last_selected_model = self._model.last_selected_model_name
    else:
      self._last_selected_model = self._config.model_name

    self._cycle_count += 1
    return self._last_prediction

  def fit_predict_secondary(self, history: pd.DataFrame) -> float:
    if self._model_secondary is None:
      return self._last_prediction_secondary
    history = _normalize_history(history, self._live_freq)
    _log_history_sample("fit_predict_secondary.history", history, self._live_freq)
    if len(history) < MIN_HISTORY_POINTS:
      fallback = _last_observed_value(history)
      self._last_prediction_secondary = max(self._last_prediction_secondary, fallback)
      return self._last_prediction_secondary

    t0_fit = time.monotonic()
    ok = _fit_with_timeout(
      self._model_secondary,
      history,
      self._live_freq,
      self._config.refit_timeout_seconds,
    )
    self.last_fit_time_secondary = time.monotonic() - t0_fit
    
    if not ok:
      self._refit_skip_count += 1
      self._last_prediction_secondary = max(
        self._last_prediction_secondary, _last_observed_value(history)
      )
      return self._last_prediction_secondary

    test_df = _make_next_timestamp_df(history, self._live_freq)
    _log_history_sample("fit_predict_secondary.test_df", test_df, self._live_freq)
    try:
      t0_pred = time.monotonic()
      pred = float(
        self._model_secondary.predict(test_df, TARGET_COL, self._live_freq).iloc[0]
      )
      self.last_predict_time_secondary = time.monotonic() - t0_pred
    except Exception as exc:
      logger.error("secondary predict() failed: %s", exc)
      self._refit_skip_count += 1
      self._last_prediction_secondary = max(
        self._last_prediction_secondary, _last_observed_value(history)
      )
      return self._last_prediction_secondary

    logger.info("fit_predict_secondary.raw_pred=%s freq=%s", pred, self._live_freq)
    self._last_prediction_secondary = max(0.0, pred)
    return self._last_prediction_secondary


  @property
  def last_selected_model(self) -> str:
    return self._last_selected_model

  @property
  def refit_skip_count(self) -> int:
    return self._refit_skip_count
