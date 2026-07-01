import os
import pandas as pd

def main():
  data_dir = "data/cegedim"
  files = [f for f in os.listdir(data_dir) if f.endswith("_unified.csv")]
  
  exp1_cutoff_date = "2022-04-10 00:00:00"
  exp2_start_date = "2022-04-10 00:00:00" 
  
  for filename in files:
    filepath = os.path.join(data_dir, filename)
    print(f"Processing {filename}...")
    
    df = pd.read_csv(filepath)
    df['timestamp_date_format'] = pd.to_datetime(df['timestamp_date_format'])
    df = df.sort_values(by="timestamp_date_format").reset_index(drop=True)
    
    exp1_df = df[df['timestamp_date_format'] < exp1_cutoff_date].copy()
    
    exp2_df = df[df['timestamp_date_format'] >= exp2_start_date].copy()
    
    base_name = filename.replace("_unified.csv", "")
    exp1_out = os.path.join(data_dir, f"{base_name}_exp1.csv")
    exp2_out = os.path.join(data_dir, f"{base_name}_exp2_replay.csv")
    
    exp1_df.to_csv(exp1_out, index=False)
    exp2_df.to_csv(exp2_out, index=False)
    
    print(f"  -> Exp 1 size: {len(exp1_df)} rows. Saved to {os.path.basename(exp1_out)}")
    print(f"  -> Exp 2 size: {len(exp2_df)} rows. Saved to {os.path.basename(exp2_out)}")

if __name__ == "__main__":
  main()
