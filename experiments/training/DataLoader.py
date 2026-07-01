import os
import pandas as pd

try:
  from constants import (
    GIGA_BYTE_SIZE_IN_BYTES,
    CEGEDIM_A_CPU_CORE_COUNT,
    CEGEDIM_A_RAM_SIZE_IN_GIGA,
    CEGEDIM_B_CPU_CORE_COUNT,
    CEGEDIM_B_RAM_SIZE_IN_GIGA,
    CEGEDIM_C_CPU_CORE_COUNT,
    CEGEDIM_C_RAM_SIZE_IN_GIGA,
  )
except ImportError:
  import sys
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  from constants import (
    GIGA_BYTE_SIZE_IN_BYTES,
    CEGEDIM_A_CPU_CORE_COUNT,
    CEGEDIM_A_RAM_SIZE_IN_GIGA,
    CEGEDIM_B_CPU_CORE_COUNT,
    CEGEDIM_B_RAM_SIZE_IN_GIGA,
    CEGEDIM_C_CPU_CORE_COUNT,
    CEGEDIM_C_RAM_SIZE_IN_GIGA,
  )

def load_dataset(filepath, dataset_type="cegedim"):
  if dataset_type == "cegedim":
    return _preprocess_cegedim_file(filepath)
  else:
    raise ValueError(f"Unknown dataset_type: {dataset_type}")

def _preprocess_cegedim_file(filepath):
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"Dataset file not found: {filepath}")

  df = pd.read_csv(filepath)
  filename = os.path.basename(filepath)

  try:
    dataset_provider, _, app_name, metric_name, dataset_type_str = filename.split("_")
  except ValueError:
    raise ValueError(f"Filename {filename} does not match expected cegedim format.")

  if app_name == "A":
    cpu_cores = CEGEDIM_A_CPU_CORE_COUNT
    ram_size = CEGEDIM_A_RAM_SIZE_IN_GIGA
  elif app_name == "B":
    cpu_cores = CEGEDIM_B_CPU_CORE_COUNT
    ram_size = CEGEDIM_B_RAM_SIZE_IN_GIGA
  elif app_name == "C":
    cpu_cores = CEGEDIM_C_CPU_CORE_COUNT
    ram_size = CEGEDIM_C_RAM_SIZE_IN_GIGA
  else:
    raise ValueError(f"Unknown app_name: {app_name}")

  metric_cols = df.columns[2:]

  if metric_name == "CPU":
    df[metric_cols] = (df[metric_cols] * cpu_cores / 100).round(2)
    freq = "1H"
  elif metric_name == "RAM":
    # App B is already in GB
    if app_name != "B":
        df[metric_cols] = df[metric_cols] / GIGA_BYTE_SIZE_IN_BYTES
    freq = "40min"
  else:
    raise ValueError(f"Unknown metric_name: {metric_name}")

  df['timestamp_date_format'] = pd.to_datetime(df['timestamp_date_format'])

  df = df.set_index('timestamp_date_format')
  df = df.sort_index()

  df['Total_Usage'] = df[metric_cols].sum(axis=1)
  df = df.drop(columns=metric_cols)
  
  df = df[['Total_Usage']].resample(freq).mean().interpolate()

  df = df.reset_index()

  train_percentage = 0.70
  val_percentage = 0.15
  test_percentage = 0.15

  total_size = len(df)
  train_size = int(total_size * train_percentage)
  val_size = int(total_size * val_percentage)

  train_df = df[0 : train_size]
  val_df   = df[train_size : train_size + val_size]
  test_df  = df[train_size + val_size : ]

  val_df = val_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  return train_df, val_df, test_df, freq