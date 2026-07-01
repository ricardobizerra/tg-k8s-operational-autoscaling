import optuna
import time
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import importlib

from training.tuning_registry import suggest_hyperparameters
from training.metrics import calculate_metrics
from training import phase_io

def instantiate_model(model_name, params, ensemble_models_instances=None):
  if model_name == "Ensemble":
    module = importlib.import_module("models.Ensemble")
    model_class = getattr(module, "Ensemble")
    return model_class(models=ensemble_models_instances, **params)

  module = importlib.import_module(f"models.{model_name}")
  model_class = getattr(module, model_name)
  return model_class(**params)

def _build_model(model_name, params, dataset_name=None, ensemble_models_names=None):
  if model_name == "Ensemble":
    ensemble_params = {}
    for key, value in params.items():
      if key.startswith("ensemble_"):
        ensemble_params[key.replace("ensemble_", "")] = value
      else:
        ensemble_params[key] = value

    inner_instances = []
    if ensemble_models_names and dataset_name:
      for name in ensemble_models_names:
        inner_run_dir = phase_io.get_latest_run_dir("results", dataset_name, name)
        inner_best_params = phase_io.load_best_params(inner_run_dir)
        inner_instances.append(instantiate_model(name, inner_best_params))

    return instantiate_model("Ensemble", ensemble_params, ensemble_models_instances=inner_instances)
  else:
    return instantiate_model(model_name, params)

def tune_model(model_name, train, val, column_name, freq, dataset_name=None, n_trials=75, ensemble_models_names=None, fixed_params=None):
  def objective(trial):
    try:
      params = suggest_hyperparameters(model_name, trial)
      if fixed_params:
        params.update(fixed_params)
      model = _build_model(model_name, params, dataset_name, ensemble_models_names)

      model.fit(train, column_name, freq)
      preds = model.predict(val, column_name, freq)

      actual_val = val[column_name]
      from sklearn.metrics import mean_squared_error
      rmse_val = mean_squared_error(actual_val, preds, squared=False)

      if np.isnan(rmse_val) or np.isinf(rmse_val):
        return 1e9

      return rmse_val
    except Exception as e:
      print(f"Trial failed with exception: {e}")
      return 1e9

  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=n_trials, n_jobs=1)
  
  best_params = study.best_params
  if fixed_params:
    best_params.update(fixed_params)

  return best_params, study

def run_data_stage(dataset_path, run_dir):
  from training.DataLoader import load_dataset

  print("\n--- [Stage: data] ---")
  train_df, val_df, test_df, freq = load_dataset(dataset_path)
  target_col = train_df.columns[1]

  print(f"  Dataset: {os.path.basename(dataset_path)}")
  print(f"  Target column: {target_col}, Freq: {freq}")
  print(f"  Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

  phase_io.save_splits(run_dir, train_df, val_df, test_df, target_col, freq)
  print("  [Stage: data] complete.")

def run_tuning_stage(run_dir, model_name, dataset_name, n_trials, ensemble_models_names=None, fixed_params=None):
  print("\n--- [Stage: tuning] ---")
  train_df, val_df, _, target_col, freq = phase_io.load_splits(run_dir)

  best_params, study = tune_model(
    model_name, train_df, val_df, target_col, freq, dataset_name,
    n_trials=n_trials, ensemble_models_names=ensemble_models_names,
    fixed_params=fixed_params
  )

  print(f"  Best params: {best_params}")
  print(f"  Best val RMSE: {study.best_value:.4f}")

  phase_io.save_tuning_artifacts(run_dir, best_params, study)

  _save_opt_history_chart(study, run_dir, model_name, target_col)
  print("  [Stage: tuning] complete.")

  return best_params, study


def run_training_stage(run_dir, model_name, dataset_name, ensemble_models_names=None):
  print("\n--- [Stage: training] ---")
  train_df, val_df, _, target_col, freq = phase_io.load_splits(run_dir)
  best_params = phase_io.load_best_params(run_dir)

  train_and_val = pd.concat([train_df, val_df], ignore_index=True)
  print(f"  Fitting on train+val ({len(train_and_val)} rows) with best params: {best_params}")

  model = _build_model(model_name, best_params, dataset_name, ensemble_models_names)

  start = time.time()
  model.fit(train_and_val, target_col, freq)
  elapsed = time.time() - start

  print(f"  Fit completed in {elapsed:.2f}s")
  print("  [Stage: training] complete. (Dry-run — no model artifact saved.)")


def run_results_stage(run_dir, model_name, dataset_name, ensemble_models_names=None):
  print("\n--- [Stage: results] ---")
  train_df, val_df, test_df, target_col, freq = phase_io.load_splits(run_dir)
  best_params = phase_io.load_best_params(run_dir)

  model = _build_model(model_name, best_params, dataset_name, ensemble_models_names)

  print(f"  Fitting on train ({len(train_df)} rows), then predicting on val ({len(val_df)} rows).")
  model.fit(train_df, target_col, freq)
  val_predictions = model.predict(val_df, target_col, freq)

  train_and_val = pd.concat([train_df, val_df], ignore_index=True)
  print(f"  Fitting on train+val ({len(train_and_val)} rows), then predicting on test ({len(test_df)} rows).")

  start_fit = time.time()
  model.fit(train_and_val, target_col, freq)
  fit_time = time.time() - start_fit

  start_pred = time.time()
  test_predictions = model.predict(test_df, target_col, freq)
  predict_time = time.time() - start_pred

  if hasattr(model, 'name') and model.name == "ensemble":
    fit_time = model.cumulative_fit_time
    predict_time = model.cumulative_predict_time
    avg_fit_time = fit_time / max(1, model.evaluations_count)
    avg_predict_time = predict_time / max(1, model.evaluations_count)
  else:
    avg_fit_time = fit_time
    avg_predict_time = predict_time

  timings = {
    "FIT_TIME_TOTAL": fit_time,
    "PREDICT_TIME_TOTAL": predict_time,
    "FIT_TIME_AVG": avg_fit_time,
    "PREDICT_TIME_AVG": avg_predict_time
  }
  
  phase_io.save_timings(run_dir, timings)
  phase_io.save_predictions(run_dir, val_predictions, test_predictions)

  print("  [Stage: results] complete.")
  return val_predictions, test_predictions


def run_evaluate_stage(run_dir, model_name, dataset_name, ensemble_models_names=None):
  print("\n--- [Stage: evaluate] ---")
  train_df, val_df, test_df, target_col, freq = phase_io.load_splits(run_dir)
  best_params = phase_io.load_best_params(run_dir)
  val_preds, test_preds = phase_io.load_predictions(run_dir)
  timings = phase_io.load_timings(run_dir)

  train_and_val = pd.concat([train_df, val_df], ignore_index=True)

  print("  Calculating Validation Metrics...")
  metrics_val = calculate_metrics(val_df[target_col], val_preds, train_df[target_col], freq)

  print("  Calculating Test Metrics...")
  metrics_test = calculate_metrics(test_df[target_col], test_preds, train_and_val[target_col], freq)
  
  for k, v in timings.items():
    metrics_test[k] = v

  print("  Test Metrics:")
  for k, v in metrics_test.items():
    print(f"    {k}: {v:.4f}")

  all_metrics = {
    "validation": metrics_val,
    "test": metrics_test
  }

  _save_metadata(run_dir, dataset_name, target_col, model_name, best_params, all_metrics)

  _save_forecast_chart(run_dir, model_name, dataset_name, target_col, val_df[target_col], val_preds, metrics_val, "val")
  _save_forecast_chart(run_dir, model_name, dataset_name, target_col, test_df[target_col], test_preds, metrics_test, "test")

  print("  [Stage: evaluate] complete.")
  return all_metrics

def _save_opt_history_chart(study, run_dir, model_name, target_col):
  from optuna.visualization.matplotlib import plot_optimization_history
  chart_dir = os.path.join(run_dir, "charts")
  os.makedirs(chart_dir, exist_ok=True)

  plt.figure(figsize=(10, 6))
  ax = plot_optimization_history(study, target_name="RMSE")
  ax.set_ylim(bottom=0)
  ax.set_xlim(left=0)
  plt.title(f"Optimization History: {model_name} - {target_col}")
  path = os.path.join(chart_dir, "opt_history.png")
  plt.savefig(path)
  plt.close()
  print(f"  Optimization history chart saved to: {path}")

def _save_forecast_chart(run_dir, model_name, dataset_name, target_col, actuals, predictions, metrics, split_name):
  chart_dir = os.path.join(run_dir, "charts")
  os.makedirs(chart_dir, exist_ok=True)

  plt.figure(figsize=(15, 7))
  plt.plot(actuals.values[:200], label="Actual", color="blue", alpha=0.7)
  plt.plot(predictions.values[:200], label="Predicted", color="red", linestyle="--", alpha=0.7)
  plt.title(
    f"Forecast vs Actual ({split_name.upper()}): {dataset_name} | Model: {model_name}\n"
    f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}"
  )
  plt.xlabel("Time steps")
  plt.ylabel("Resource Usage")
  plt.legend()
  plt.grid(True, alpha=0.3)

  path = os.path.join(chart_dir, f"forecast_vs_actual_{split_name}.png")
  plt.savefig(path)
  plt.close()
  print(f"  Forecast chart ({split_name}) saved to: {path}")

def _save_metadata(run_dir, dataset_name, target_col, model_name, best_params, metrics):
  metadata = {
    "dataset": dataset_name,
    "target_col": target_col,
    "model_type": model_name,
    "best_params": best_params,
    "metrics": metrics,
  }
  path = os.path.join(run_dir, "metadata.json")
  with open(path, "w") as f:
    json.dump(metadata, f, indent=4)
  print(f"  Metadata saved to: {path}")

def aggregate_results(results_dir="results", run_timestamp=None):
  import glob as glob_mod

  if run_timestamp:
    metadata_files = glob_mod.glob(os.path.join(results_dir, "**", f"*{run_timestamp}*", "metadata.json"), recursive=True)
    # Legacy fallback
    metadata_files += glob_mod.glob(os.path.join(results_dir, f"*_{run_timestamp}_metadata.json"))
    summary_path = os.path.join(results_dir, f"results_summary_{run_timestamp}.csv")
  else:
    metadata_files = glob_mod.glob(os.path.join(results_dir, "**", "metadata.json"), recursive=True)
    # Legacy fallback
    metadata_files += glob_mod.glob(os.path.join(results_dir, "*_metadata.json"))
    summary_path = os.path.join(results_dir, "results_summary.csv")

  rows = []
  for file in metadata_files:
    with open(file, "r") as f:
      data = json.load(f)
      row = {
        "Dataset": data.get("dataset"),
        "Target": data.get("target_col"),
        "Model": data.get("model_type"),
      }
      metrics = data.get("metrics", {})
      test_metrics = metrics.get("test", metrics) # fallback to top-level if old format
      for k, v in test_metrics.items():
        row[k] = v
      rows.append(row)

  df = pd.DataFrame(rows)
  if not df.empty:
    df.to_csv(summary_path, index=False)
    print(f"Aggregated summary saved to: {summary_path}")
  return df