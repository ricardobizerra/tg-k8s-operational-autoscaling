import os
import json
import glob
import pandas as pd

def get_run_dir(results_dir, dataset_name, model_name, timestamp):
  return os.path.join(results_dir, dataset_name, model_name, timestamp)

def get_latest_run_dir(results_dir, dataset_name, model_name):
  search_path = os.path.join(results_dir, dataset_name, model_name)
  if not os.path.isdir(search_path):
    raise FileNotFoundError(
      f"No prior run found for dataset='{dataset_name}' model='{model_name}'. "
      f"Expected directory: {search_path}\n"
      f"Hint: run '--stages tuning' first to generate a tuning result."
    )

  subdirs = sorted([
    d for d in os.listdir(search_path)
    if os.path.isdir(os.path.join(search_path, d))
  ])

  if not subdirs:
    raise FileNotFoundError(
      f"No timestamp subdirectories found in {search_path}.\n"
      f"Hint: run '--stages tuning' first to generate a tuning result."
    )

  latest = subdirs[-1]
  return os.path.join(search_path, latest)

def get_run_dir_from_timestamp(results_dir, dataset_name, model_name, from_timestamp):
  run_dir = get_run_dir(results_dir, dataset_name, model_name, from_timestamp)
  if not os.path.isdir(run_dir):
    raise FileNotFoundError(
      f"No run directory found for timestamp '{from_timestamp}': {run_dir}"
    )
  return run_dir

_SPLITS_META_FILENAME = "splits_meta.json"
_TRAIN_FILENAME = "train.csv"
_VAL_FILENAME = "val.csv"
_TEST_FILENAME = "test.csv"

def save_splits(run_dir, train_df, val_df, test_df, target_col, freq):
  os.makedirs(run_dir, exist_ok=True)

  train_df.to_csv(os.path.join(run_dir, _TRAIN_FILENAME), index=False)
  val_df.to_csv(os.path.join(run_dir, _VAL_FILENAME), index=False)
  test_df.to_csv(os.path.join(run_dir, _TEST_FILENAME), index=False)

  train_min = float(train_df[target_col].min())
  train_max = float(train_df[target_col].max())
  train_mean = float(train_df[target_col].mean())
  train_std = float(train_df[target_col].std())

  meta = {
      "target_col": target_col,
      "freq": freq,
      "train_min": train_min,
      "train_max": train_max,
      "train_mean": train_mean,
      "train_std": train_std
  }
  with open(os.path.join(run_dir, _SPLITS_META_FILENAME), "w") as f:
    json.dump(meta, f, indent=2)

  print(f"  Splits saved to: {run_dir}")
  print(f"    train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
  print(f"    target_col={target_col}, freq={freq}")

def load_splits(run_dir):
  _require_file(run_dir, _SPLITS_META_FILENAME, "data")
  _require_file(run_dir, _TRAIN_FILENAME, "data")
  _require_file(run_dir, _VAL_FILENAME, "data")
  _require_file(run_dir, _TEST_FILENAME, "data")

  with open(os.path.join(run_dir, _SPLITS_META_FILENAME)) as f:
    meta = json.load(f)

  train_df = pd.read_csv(os.path.join(run_dir, _TRAIN_FILENAME))
  val_df = pd.read_csv(os.path.join(run_dir, _VAL_FILENAME))
  test_df = pd.read_csv(os.path.join(run_dir, _TEST_FILENAME))

  return train_df, val_df, test_df, meta["target_col"], meta["freq"]

_TUNING_FILENAME = "tuning.json"

def save_tuning_artifacts(run_dir, best_params, study):
  os.makedirs(run_dir, exist_ok=True)

  trials_dict = study.trials_dataframe().to_dict(orient="records")
  payload = {
    "best_params": best_params,
    "best_rmse_val": study.best_value,
    "trials": trials_dict,
  }

  tuning_path = os.path.join(run_dir, _TUNING_FILENAME)
  with open(tuning_path, "w") as f:
    json.dump(payload, f, indent=4, default=str)

  print(f"  Tuning artifacts saved to: {tuning_path}")

def save_fixed_params(run_dir, params):
  os.makedirs(run_dir, exist_ok=True)
  payload = {"best_params": params, "best_rmse_val": None, "trials": []}
  path = os.path.join(run_dir, _TUNING_FILENAME)
  with open(path, "w") as f:
    json.dump(payload, f, indent=4)
  print(f"  Fixed params saved to: {path}")

def load_best_params(run_dir):
  _require_file(run_dir, _TUNING_FILENAME, "tuning")

  with open(os.path.join(run_dir, _TUNING_FILENAME)) as f:
    payload = json.load(f)

  return payload["best_params"]

_PREDS_VAL_FILENAME = "predictions_val.csv"
_PREDS_TEST_FILENAME = "predictions_test.csv"

def save_predictions(run_dir, val_preds, test_preds):
  os.makedirs(run_dir, exist_ok=True)
  val_preds.to_csv(os.path.join(run_dir, _PREDS_VAL_FILENAME), index=False)
  test_preds.to_csv(os.path.join(run_dir, _PREDS_TEST_FILENAME), index=False)
  print(f"  Predictions saved to: {run_dir}")

def load_predictions(run_dir):
  _require_file(run_dir, _PREDS_VAL_FILENAME, "results")
  _require_file(run_dir, _PREDS_TEST_FILENAME, "results")
  val_preds = pd.read_csv(os.path.join(run_dir, _PREDS_VAL_FILENAME)).iloc[:, 0]
  test_preds = pd.read_csv(os.path.join(run_dir, _PREDS_TEST_FILENAME)).iloc[:, 0]
  return val_preds, test_preds

_TIMINGS_FILENAME = "timings.json"

def save_timings(run_dir, timings):
  with open(os.path.join(run_dir, _TIMINGS_FILENAME), "w") as f:
    json.dump(timings, f, indent=4)

def load_timings(run_dir):
  _require_file(run_dir, _TIMINGS_FILENAME, "results")
  with open(os.path.join(run_dir, _TIMINGS_FILENAME)) as f:
    return json.load(f)

def _require_file(run_dir, filename, producing_stage):
  path = os.path.join(run_dir, filename)
  if not os.path.exists(path):
    raise FileNotFoundError(
      f"Required file '{filename}' not found in run directory: {run_dir}\n"
      f"Hint: run '--stages {producing_stage}' first to produce this file."
    )
