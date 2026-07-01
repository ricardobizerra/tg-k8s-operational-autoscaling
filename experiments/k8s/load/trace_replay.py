import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import constants

GIGA_BYTE_SIZE_IN_BYTES = 1_073_741_824

APP_CPU_CORES = {"A": 4, "B": 4, "C": 2}
APP_RAM_SIZE_GB = {"A": 32, "B": 16, "C": 4}

# Apps whose RAM column is already in GB (no byte→GB conversion needed)
RAM_ALREADY_GB = {"B"}


def _infer_app_and_metric(filepath: str):
  filename = os.path.basename(filepath)
  parts = filename.replace(".csv", "").split("_")
  if len(parts) < 4 or parts[0] != "cegedim" or parts[1] != "application":
    raise ValueError(
      f"Filename '{filename}' does not match expected pattern "
      "cegedim_application_{{APP}}_{{METRIC}}*.csv"
    )
  app = parts[2].upper()
  metric = parts[3].upper()
  if app not in APP_CPU_CORES:
    raise ValueError(f"Unknown app '{app}'; expected one of {list(APP_CPU_CORES)}")
  if metric not in ("CPU", "RAM"):
    raise ValueError(f"Unknown metric '{metric}'; expected CPU or RAM")
  return app, metric


def _load_and_preprocess(filepath: str, app: str, metric: str) -> pd.DataFrame:
  df = pd.read_csv(filepath)

  metric_cols = df.columns[2:]

  if metric == "CPU":
    cpu_cores = APP_CPU_CORES[app]
    df[metric_cols] = (df[metric_cols] * cpu_cores / 100).round(2)
    freq = "1H"
  elif metric == "RAM":
    if app not in RAM_ALREADY_GB:
      df[metric_cols] = df[metric_cols] / GIGA_BYTE_SIZE_IN_BYTES
    freq = "40min"

  df["timestamp_date_format"] = pd.to_datetime(df["timestamp_date_format"])
  df = df.set_index("timestamp_date_format").sort_index()

  df["Total_Usage"] = df[metric_cols].sum(axis=1)
  df = df[["Total_Usage"]].resample(freq).mean().interpolate()
  df = df.reset_index()

  return df


def _build_cpu_trace(df: pd.DataFrame, app: str, peak_rps: int, warmup_steps: int) -> list:
  app_max_cpu_cores = float(constants.APP_TOTAL_CPU_CORES[app])
  records = []
  for i, row in df.iterrows():
    cpu_cores = float(row["Total_Usage"])
    ratio = cpu_cores / app_max_cpu_cores
    rps = max(1, round(ratio * peak_rps))
    records.append(
      {
        "index": int(i),
        "timestamp": row["timestamp_date_format"].isoformat(),
        "rps": rps,
        "cpu_actual_cores": round(cpu_cores, 4),
        "phase": "warmup" if i < warmup_steps else "evaluation",
      }
    )
  return records


def _build_ram_trace(
  df: pd.DataFrame,
  app: str,
  min_memory_mb: int,
  max_memory_mb: int,
  memory_step_mb: int,
  warmup_steps: int,
) -> list:
  app_max_ram_gb = float(constants.APP_TOTAL_RAM_SIZE_GB[app])
    
  records = []
  for i, row in df.iterrows():
    ram_gb = float(row["Total_Usage"])
    
    ratio = ram_gb / app_max_ram_gb
    memory_mb = round(ratio * max_memory_mb)
    
    memory_mb = max(min_memory_mb, min(memory_mb, max_memory_mb))
    memory_mb = round(memory_mb / memory_step_mb) * memory_step_mb
    records.append(
      {
        "index": int(i),
        "timestamp": row["timestamp_date_format"].isoformat(),
        "memory_mb": memory_mb,
        "ram_actual_gb": round(ram_gb, 4),
        "phase": "warmup" if i < warmup_steps else "evaluation",
      }
    )
  return records


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description="Convert a cegedim unified CSV into a JSON trace for Locust / ram_injector.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--csv", required=True, help="Path to unified cegedim CSV")
  parser.add_argument(
    "--metric",
    required=True,
    choices=["cpu", "ram"],
    help="Metric to convert",
  )
  parser.add_argument(
    "--peak-rps",
    type=int,
    default=None,
    help="Max RPS calibrated for the app (required for --metric cpu)",
  )
  parser.add_argument(
    "--min-memory-mb",
    type=int,
    default=64,
    help="Minimum RAM buffer size in MB (RAM traces only)",
  )
  parser.add_argument(
    "--max-memory-mb",
    type=int,
    default=8192,
    help="Maximum RAM buffer size in MB (RAM traces only)",
  )
  parser.add_argument(
    "--memory-step-mb",
    type=int,
    default=64,
    help="RAM rounding step in MB (RAM traces only)",
  )
  parser.add_argument(
    "--warmup-steps",
    type=int,
    default=0,
    help="Number of initial points to mark as phase='warmup'",
  )
  parser.add_argument(
    "--output-dir",
    default="k8s/load/traces/",
    help="Directory to write the output JSON trace",
  )
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)

  if args.metric == "cpu":
    if args.peak_rps is None:
      print(
        "ERROR: --peak-rps is required for --metric cpu",
        file=sys.stderr,
      )
      sys.exit(1)

  app, metric_from_file = _infer_app_and_metric(args.csv)
  if metric_from_file != args.metric.upper():
    print(
      f"WARNING: filename implies metric '{metric_from_file}' "
      f"but --metric '{args.metric.upper()}' was given. Proceeding with --metric.",
      file=sys.stderr,
    )

  print(f"Loading: {args.csv}")
  df = _load_and_preprocess(args.csv, app, args.metric.upper())

  df = df.reset_index(drop=True)

  if args.metric == "cpu":
    trace = _build_cpu_trace(df, app, args.peak_rps, args.warmup_steps)
  else:
    trace = _build_ram_trace(
      df, app, args.min_memory_mb, args.max_memory_mb, args.memory_step_mb, args.warmup_steps
    )

  output_name = f"{app.lower()}-{args.metric.lower()}_trace.json"
  os.makedirs(args.output_dir, exist_ok=True)
  output_path = os.path.join(args.output_dir, output_name)

  with open(output_path, "w") as f:
    json.dump(trace, f, indent=2)

  print(f"Wrote {len(trace)} trace points → {output_path}")


if __name__ == "__main__":
  main()
