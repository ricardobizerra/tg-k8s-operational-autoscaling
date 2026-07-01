import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

try:
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy.stats import wilcoxon
except ImportError as e:
  print(f"Error: Missing analysis dependencies ({e})", file=sys.stderr)
  print("Please install them before running this script:", file=sys.stderr)
  print("  pip install scipy seaborn matplotlib pandas numpy", file=sys.stderr)
  sys.exit(1)

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger("analyze_results")

SLUG_TO_DATASET_NAME = {
  "a-cpu": "cegedim_application_A_CPU_exp1",
  "a-ram": "cegedim_application_A_RAM_exp1",
  "b-cpu": "cegedim_application_B_CPU_exp1",
  "b-ram": "cegedim_application_B_RAM_exp1",
  "c-cpu": "cegedim_application_C_CPU_exp1",
  "c-ram": "cegedim_application_C_RAM_exp1",
}


def _load_exp1_metadata(exp1_results_dir: Path, dataset_slug: str, model_name: str) -> Optional[dict]:
  dataset_name = SLUG_TO_DATASET_NAME.get(dataset_slug)
  if not dataset_name:
    return None

  search_path = exp1_results_dir / dataset_name / model_name
  if not search_path.is_dir():
    return None

  subdirs = sorted([d for d in search_path.iterdir() if d.is_dir()])
  if not subdirs:
    return None
  latest_run_dir = subdirs[-1]

  metadata_path = latest_run_dir / "metadata.json"
  if not metadata_path.exists():
    return None

  try:
    with open(metadata_path, "r") as f:
      return json.load(f)
  except Exception:
    return None


def _get_exp1_test_metrics(metadata: dict) -> Tuple[float, float, float]:
  metrics = metadata.get("metrics", {})
  test_metrics = metrics.get("test", metrics)
  return (
    test_metrics.get("MAE", np.nan),
    test_metrics.get("RMSE", np.nan),
    test_metrics.get("R2", np.nan),
  )

def build_summary_all(exp2_results_dir: Path, output_dir: Path) -> pd.DataFrame:
  rows = []
  
  for summary_path in exp2_results_dir.rglob("summary.json"):
    if "analysis" in summary_path.parts:
      continue
      
    with open(summary_path, "r") as f:
      try:
        data = json.load(f)
      except Exception as e:
        logger.warning("Failed to parse %s: %s", summary_path, e)
        continue

    run_id = summary_path.parent.parent.name
        
    pm = data.get("prediction_metrics", {})
    cpu_pm = pm.get("cpu", {})
    ram_pm = pm.get("ram", {})
    om = data.get("operational_metrics", {})
    
    row = {
      "run_id": run_id,
      "model_name": data.get("model_name"),
      "dataset_slug": data.get("dataset_slug"),
      "scaling_mode": data.get("scaling_mode"),
      "namespace": data.get("namespace"),
      
      "MAE_cpu": cpu_pm.get("MAE"),
      "RMSE_cpu": cpu_pm.get("RMSE"),
      "R2_cpu": cpu_pm.get("R2"),
      
      "MAE_ram": ram_pm.get("MAE"),
      "RMSE_ram": ram_pm.get("RMSE"),
      "R2_ram": ram_pm.get("R2"),
      
      "sla_cpu_violation_rate": om.get("sla_cpu_violation_rate"),
      "sla_ram_violation_rate": om.get("sla_ram_violation_rate"),
      "overprovision_cpu_minutes": om.get("overprovision_cpu_minutes"),
      "overprovision_ram_minutes": om.get("overprovision_ram_minutes"),
      "oscillation_horizontal_count": om.get("oscillation_horizontal_count"),
      "oscillation_vertical_count": om.get("oscillation_vertical_count"),
      "scale_up_horizontal_count": om.get("scale_up_horizontal_count"),
      "scale_down_horizontal_count": om.get("scale_down_horizontal_count"),
      "scale_up_vertical_count": om.get("scale_up_vertical_count"),
      "scale_down_vertical_count": om.get("scale_down_vertical_count"),
      "avg_replicas": om.get("avg_replicas"),
      "avg_memory_mb": om.get("avg_memory_mb"),
      "refit_skip_count": om.get("refit_skip_count"),
      "avg_fit_time_cpu_seconds": om.get("avg_fit_time_cpu_seconds"),
      "max_fit_time_cpu_seconds": om.get("max_fit_time_cpu_seconds"),
      "avg_predict_time_cpu_seconds": om.get("avg_predict_time_cpu_seconds"),
      "max_predict_time_cpu_seconds": om.get("max_predict_time_cpu_seconds"),
      "avg_fit_time_ram_seconds": om.get("avg_fit_time_ram_seconds"),
      "max_fit_time_ram_seconds": om.get("max_fit_time_ram_seconds"),
      "avg_predict_time_ram_seconds": om.get("avg_predict_time_ram_seconds"),
      "max_predict_time_ram_seconds": om.get("max_predict_time_ram_seconds"),
    }
    rows.append(row)
    
  df = pd.DataFrame(rows)
  if not df.empty:
    df.to_csv(output_dir / "summary_all.csv", index=False)
    logger.info("Created summary_all.csv (%d rows)", len(df))
  else:
    logger.warning("No Exp2 summaries found to create summary_all.csv")
    
  return df


def build_comparison(
  summary_df: pd.DataFrame, 
  exp1_results_dir: Path, 
  output_dir: Path
) -> pd.DataFrame:
  if summary_df.empty:
    return pd.DataFrame()
    
  rows = []
  
  for _, row in summary_df.iterrows():
    model_name = row["model_name"]
    dataset_slug = row["dataset_slug"]
    scaling_mode = row["scaling_mode"]
    
    if dataset_slug.endswith("-cpu"):
      mae_exp2 = row["MAE_cpu"]
      rmse_exp2 = row["RMSE_cpu"]
      r2_exp2 = row["R2_cpu"]
    else:
      mae_exp2 = row["MAE_ram"]
      rmse_exp2 = row["RMSE_ram"]
      r2_exp2 = row["R2_ram"]
      
    metadata = _load_exp1_metadata(exp1_results_dir, dataset_slug, model_name)
    if not metadata:
      logger.debug("No Exp1 metadata for %s / %s", dataset_slug, model_name)
      continue
      
    mae_exp1, rmse_exp1, r2_exp1 = _get_exp1_test_metrics(metadata)
    
    mae_deg = np.nan
    if mae_exp1 and not np.isnan(mae_exp1) and mae_exp1 > 0:
      mae_deg = ((mae_exp2 - mae_exp1) / mae_exp1) * 100
      
    r2_deg = np.nan
    if r2_exp1 and not np.isnan(r2_exp1) and r2_exp1 > 0:
      r2_deg = ((r2_exp1 - r2_exp2) / r2_exp1) * 100
      
    rows.append({
      "model_name": model_name,
      "dataset_slug": dataset_slug,
      "scaling_mode": scaling_mode,
      "MAE_exp1_test": mae_exp1,
      "MAE_exp2": mae_exp2,
      "RMSE_exp1_test": rmse_exp1,
      "RMSE_exp2": rmse_exp2,
      "R2_exp1_test": r2_exp1,
      "R2_exp2": r2_exp2,
      "mae_degradation_pct": mae_deg,
      "r2_degradation_pct": r2_deg,
    })
    
  df = pd.DataFrame(rows)
  if not df.empty:
    df.to_csv(output_dir / "comparison_exp1_vs_exp2.csv", index=False)
    logger.info("Created comparison_exp1_vs_exp2.csv (%d matched rows)", len(df))
    
  return df


def build_statistical_tests(summary_df: pd.DataFrame, output_dir: Path) -> None:
  if summary_df.empty:
    return
    
  rows = []
  
  individuals = ["ExponentialSmoothing", "Prophet", "FFT"]
  
  metrics_to_test = [
    "MAE_cpu", "MAE_ram", 
    "sla_cpu_violation_rate", "sla_ram_violation_rate",
    "overprovision_cpu_minutes", "overprovision_ram_minutes"
  ]
  
  groups = summary_df.groupby(["dataset_slug", "scaling_mode"])
  
  for (dataset, mode), group_df in groups:
    ens_rows = group_df[group_df["model_name"].str.startswith("Ensemble")]
    if ens_rows.empty:
      continue
      
    for _, ens_row in ens_rows.iterrows():
      ens_model_name = ens_row["model_name"]
      
      for ind_model in individuals:
        if ind_model not in ens_model_name:
          continue
          
        ind_rows = group_df[group_df["model_name"] == ind_model]
        if ind_rows.empty:
          continue
          
        ind_row = ind_rows.iloc[0]
        
        for metric in metrics_to_test:
          val_ens = ens_row[metric]
          val_ind = ind_row[metric]
          
          if pd.isna(val_ens) or pd.isna(val_ind):
            continue
            
          rows.append({
            "dataset_slug": dataset,
            "scaling_mode": mode,
            "model_ensemble": ens_model_name,
            "model_individual": ind_model,
            "metric": metric,
            "val_ensemble": val_ens,
            "val_individual": val_ind
          })
        
  if not rows:
    logger.info("Not enough data to run statistical tests")
    return
    
  pairs_df = pd.DataFrame(rows)
  
  results = []
  
  for metric in metrics_to_test:
    metric_df = pairs_df[pairs_df["metric"] == metric]
    
    test_groups = metric_df.groupby(["model_ensemble", "model_individual"])
    
    for (ens_name, ind_name), subset in test_groups:
      if len(subset) < 3:
        continue
        
      ens_vals = subset["val_ensemble"].values
      ind_vals = subset["val_individual"].values
      
      diffs = ens_vals - ind_vals
      if np.all(diffs == 0):
        p_value = 1.0
      else:
        try:
          stat, p_value = wilcoxon(ind_vals, ens_vals)
        except ValueError:
          p_value = np.nan
          
      results.append({
        "group": "statistical",
        "model_individual": ind_name,
        "model_ensemble": ens_name,
        "dataset_slug": "ALL",
        "scaling_mode": "ALL",
        "metric": metric,
        "p_value": p_value,
        "significant": p_value < 0.05 if not pd.isna(p_value) else False,
        "mean_individual": np.mean(ind_vals),
        "mean_ensemble": np.mean(ens_vals),
        "n_samples": len(subset)
      })
      
  res_df = pd.DataFrame(results)
  if not res_df.empty:
    res_df.to_csv(output_dir / "statistical_tests.csv", index=False)
    logger.info("Created statistical_tests.csv (%d tests run)", len(res_df))


def plot_charts(
  summary_df: pd.DataFrame, 
  comp_df: pd.DataFrame, 
  exp2_results_dir: Path, 
  charts_dir: Path
) -> None:
  cpu_sla_df = summary_df.dropna(subset=["sla_cpu_violation_rate"])
  if not cpu_sla_df.empty:
    pivot_cpu = cpu_sla_df.pivot_table(
      index="model_name", 
      columns="dataset_slug", 
      values="sla_cpu_violation_rate", 
      aggfunc="mean"
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_cpu, annot=True, cmap="YlOrRd", fmt=".3f")
    plt.title("SLA CPU Violation Rate Heatmap")
    plt.tight_layout()
    plt.savefig(charts_dir / "sla_violations_heatmap_cpu.png")
    plt.close()
    
  ram_sla_df = summary_df.dropna(subset=["sla_ram_violation_rate"])
  if not ram_sla_df.empty:
    pivot_ram = ram_sla_df.pivot_table(
      index="model_name", 
      columns="dataset_slug", 
      values="sla_ram_violation_rate", 
      aggfunc="mean"
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_ram, annot=True, cmap="YlOrRd", fmt=".3f")
    plt.title("SLA RAM Violation Rate Heatmap")
    plt.tight_layout()
    plt.savefig(charts_dir / "sla_violations_heatmap_ram.png")
    plt.close()
    
  if "oscillation_horizontal_count" in summary_df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(
      data=summary_df, 
      x="model_name", 
      y="oscillation_horizontal_count", 
      hue="scaling_mode"
    )
    plt.title("Horizontal Oscillations by Model and Scaling Mode")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(charts_dir / "oscillations_by_model.png")
    plt.close()

  if "oscillation_vertical_count" in summary_df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(
      data=summary_df,
      x="model_name",
      y="oscillation_vertical_count",
      hue="scaling_mode"
    )
    plt.title("Vertical Oscillations by Model and Scaling Mode")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(charts_dir / "oscillations_vertical_by_model.png")
    plt.close()
    
  if "overprovision_ram_minutes" in summary_df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(
      data=summary_df,
      x="model_name",
      y="overprovision_ram_minutes",
      hue="scaling_mode"
    )
    plt.title("Overprovisioned RAM (Minutes) by Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(charts_dir / "overprovision_ram_minutes.png")
    plt.close()
    
  if not comp_df.empty and "MAE_exp1_test" in comp_df.columns:
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
      data=comp_df,
      x="MAE_exp1_test",
      y="MAE_exp2",
      hue="model_name",
      s=100
    )
    
    max_val = max(comp_df["MAE_exp1_test"].max(), comp_df["MAE_exp2"].max())
    if not pd.isna(max_val):
      plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal (No degradation)')
      
    plt.title("MAE: Experiment 1 (Test) vs Experiment 2")
    plt.xlabel("Exp 1 MAE (Static Test Split)")
    plt.ylabel("Exp 2 MAE (Online Rolling)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(charts_dir / "mae_exp1_vs_exp2_scatter.png")
    plt.close()
    
  for csv_path in exp2_results_dir.rglob("scaling_events.csv"):
    if "analysis" in csv_path.parts:
      continue
      
    try:
      events_df = pd.read_csv(csv_path)
      events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
      events_df = events_df.sort_values("timestamp")
    except Exception as e:
      logger.warning("Failed to plot timeseries for %s: %s", csv_path, e)
      continue
      
    if events_df.empty:
      continue
      
    run_id = csv_path.parent.parent.name
    ns_name = csv_path.parent.name
      
    if events_df.iloc[0]["scaling_mode"] in ("horizontal", "combined"):
      fig, ax1 = plt.subplots(figsize=(12, 6))
      
      ax1.step(events_df["timestamp"], events_df["current_replicas"], where='post', color='blue', label='Replicas', linewidth=2)
      ax1.set_xlabel('Time')
      ax1.set_ylabel('Replicas', color='blue')
      ax1.tick_params(axis='y', labelcolor='blue')
      
      ax2 = ax1.twinx()
      ax2.plot(events_df["timestamp"], events_df["actual_cpu_cores"], color='red', alpha=0.6, label='Actual CPU')
      ax2.plot(events_df["timestamp"], events_df["predicted_cpu_cores"], color='orange', alpha=0.8, linestyle='--', label='Predicted CPU')
      ax2.set_ylabel('CPU Cores', color='red')
      ax2.tick_params(axis='y', labelcolor='red')
      
      fig.suptitle(f"Replicas & CPU Over Time: {ns_name} ({run_id})")
      fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
      fig.tight_layout()
      fig.savefig(charts_dir / f"replicas_over_time_{run_id}_{ns_name}.png")
      plt.close(fig)
      
    if events_df.iloc[0]["scaling_mode"] in ("vertical", "combined"):
      fig, ax1 = plt.subplots(figsize=(12, 6))
      
      ax1.step(events_df["timestamp"], events_df["current_memory_mb"], where='post', color='purple', label='Memory Limit (MB)', linewidth=2)
      ax1.set_xlabel('Time')
      ax1.set_ylabel('Memory Limit (MB)', color='purple')
      ax1.tick_params(axis='y', labelcolor='purple')
      
      ax2 = ax1.twinx()
      ax2.plot(events_df["timestamp"], events_df["actual_ram_gb"] * 1024, color='green', alpha=0.6, label='Actual RAM (MB)')
      ax2.plot(events_df["timestamp"], events_df["predicted_ram_gb"] * 1024, color='lime', alpha=0.8, linestyle='--', label='Predicted RAM (MB)')
      ax2.set_ylabel('RAM Working Set (MB)', color='green')
      ax2.tick_params(axis='y', labelcolor='green')
      
      fig.suptitle(f"Memory Limit & RAM Over Time: {ns_name} ({run_id})")
      fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
      fig.tight_layout()
      fig.savefig(charts_dir / f"memory_over_time_{run_id}_{ns_name}.png")
      plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description="Analyze Phase 2 Experiments")
  parser.add_argument("--exp2-results-dir", default="results/exp2/")
  parser.add_argument("--exp1-results-dir", default="results/")
  parser.add_argument("--output-dir", default="results/exp2/analysis/")
  args = parser.parse_args()
  
  exp2_dir = Path(args.exp2_results_dir)
  exp1_dir = Path(args.exp1_results_dir)
  out_dir = Path(args.output_dir)
  charts_dir = out_dir / "charts"
  
  if not exp2_dir.exists():
    logger.error("Exp2 results directory not found: %s", exp2_dir)
    sys.exit(1)
    
  out_dir.mkdir(parents=True, exist_ok=True)
  charts_dir.mkdir(parents=True, exist_ok=True)
  
  logger.info("Starting analysis of %s", exp2_dir)
  
  summary_df = build_summary_all(exp2_dir, out_dir)
  
  if summary_df.empty:
    logger.warning("No data found for analysis.")
    return
    
  comp_df = build_comparison(summary_df, exp1_dir, out_dir)
  
  build_statistical_tests(summary_df, out_dir)
  
  logger.info("Generating charts in %s...", charts_dir)
  plot_charts(summary_df, comp_df, exp2_dir, charts_dir)
  
  logger.info("Analysis complete! Results saved to %s", out_dir)


if __name__ == "__main__":
  main()
