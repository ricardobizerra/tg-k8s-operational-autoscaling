import pandas as pd
import os
from constants import (
  GIGA_BYTE_SIZE_IN_BYTES,
  CEGEDIM_A_CPU_CORE_COUNT,
  CEGEDIM_A_RAM_SIZE_IN_GIGA,
  CEGEDIM_B_CPU_CORE_COUNT,
  CEGEDIM_B_RAM_SIZE_IN_GIGA,
  CEGEDIM_C_CPU_CORE_COUNT,
  CEGEDIM_C_RAM_SIZE_IN_GIGA,
)

def load_cegedim_datasets():
  datasets_path = "../datasets/cegedim/"

  directories = os.listdir(datasets_path)
  print(directories)

  for dir in directories:
    sub_directories = os.listdir(datasets_path + dir)
    print(sub_directories)

    for sub_dir in sub_directories:
      files = os.listdir(datasets_path + dir + "/" + sub_dir)
      print(files)

      output_path = "data/cegedim/" + "cegedim_" + dir + "_" + sub_dir + "_unified.csv"
      
      if os.path.exists(output_path):
        os.remove(output_path)

      df_list = []
      for file in files:
        if file.endswith(".csv"):
          filepath = datasets_path + dir + "/" + sub_dir + "/" + file
          print(filepath)
          dataset = pd.read_csv(filepath)
          df_list.append(dataset)

      if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df['timestamp_date_format'] = pd.to_datetime(combined_df['timestamp_date_format'])
        combined_df = combined_df.sort_values('timestamp_date_format')
        combined_df.to_csv(output_path, index=False)

def preprocess_cegedim_datasets():
  datasets_path = "data/cegedim/"
  processed_dfs = dict()
  for file in os.listdir(datasets_path):
    train_df, val_df, test_df = preprocess_cegedim_file(file)
    processed_dfs[file] = {
      "train": train_df,
      "val": val_df,
      "test": test_df,
    }
  
  return processed_dfs

def preprocess_cegedim_file(file):
  datasets_path = "data/cegedim/"
  if not file.endswith("_unified.csv"):
    return None
  
  filepath = datasets_path + file
  df = pd.read_csv(filepath)

  dataset_provider, _, app_name, metric_name, dataset_type = file.split("_")
  dataset_type = dataset_type.replace(".csv", "")

  if app_name == "A":
    cpu_cores = CEGEDIM_A_CPU_CORE_COUNT
    ram_size = CEGEDIM_A_RAM_SIZE_IN_GIGA
  elif app_name == "B":
    cpu_cores = CEGEDIM_B_CPU_CORE_COUNT
    ram_size = CEGEDIM_B_RAM_SIZE_IN_GIGA
  elif app_name == "C":
    cpu_cores = CEGEDIM_C_CPU_CORE_COUNT
    ram_size = CEGEDIM_C_RAM_SIZE_IN_GIGA

  metric_cols = df.columns[2:]

  if metric_name == "CPU":
    df[metric_cols] = (df[metric_cols] * cpu_cores / 100).round(2)

    has_missing = df.isnull().values.any()
    print(f"Has missing values: {has_missing}")

  elif metric_name == "RAM":
    # App B is already in GB
    if "application_B" not in dataset_name:
        df[metric_cols] = df[metric_cols] / GIGA_BYTE_SIZE_IN_BYTES

    has_missing = df.isnull().values.any()
    print(f"Has missing values: {has_missing}")

  if metric_name == "CPU":
    freq = "1H"
  elif metric_name == "RAM":
    freq = "40min"
  
  df['timestamp_date_format'] = pd.to_datetime(df['timestamp_date_format'])

  df['Total_Usage'] = df[metric_cols].sum(axis=1)
  df.drop(columns=metric_cols, inplace=True)
  metric_cols = ['Total_Usage']

  df = df.set_index('timestamp_date_format')

  df = df[metric_cols].resample(freq).mean().interpolate()

  df = df.reset_index()

  train_percentage = 0.70
  val_percentage = 0.15
  test_percentage = 0.15

  if train_percentage + val_percentage + test_percentage != 1.0:
    raise ValueError("The sum of train, validation, and test percentages must be 1.0")

  total_size = len(df)
  train_size = int(total_size * train_percentage)
  val_size = int(total_size * val_percentage)

  train_df = df[0 : train_size]
  val_df   = df[train_size : train_size + val_size]
  test_df  = df[train_size + val_size : ]

  val_df = val_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  print(f"Train size: {len(train_df)}")
  print(f"Validation size: {len(val_df)}")
  print(f"Test size: {len(test_df)}")

  return train_df, val_df, test_df, freq

        
