import numpy as np
from sklearn.metrics import r2_score

def infer_seasonal_lag(freq):
  if freq in ['1H', 'H']:
    return 24
  elif freq in ['40T', '40min']:
    return 36
  return 1

def calculate_metrics(y_true, y_pred, y_train, freq):
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  y_train = np.asarray(y_train)

  errors = y_true - y_pred
  abs_errors = np.abs(errors)
  squared_errors = errors ** 2

  mae = np.mean(abs_errors)

  rmse = np.sqrt(np.mean(squared_errors))

  denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
  smape = np.mean(abs_errors / np.where(denominator == 0, 1e-8, denominator)) * 100

  m = infer_seasonal_lag(freq)
  if len(y_train) > m:
    naive_errors = y_train[m:] - y_train[:-m]
    naive_mae = np.mean(np.abs(naive_errors))
    naive_rmse = np.sqrt(np.mean(naive_errors ** 2))

    mase = mae / naive_mae if naive_mae > 0 else np.nan
    rmsse = rmse / naive_rmse if naive_rmse > 0 else np.nan
  else:
    mase = np.nan
    rmsse = np.nan

  r2 = r2_score(y_true, y_pred)

  mean_true = np.mean(y_true)
  nmae = mae / mean_true if mean_true > 0 else np.nan

  nrmse = rmse / mean_true if mean_true > 0 else np.nan

  std_errors = np.std(errors)

  return {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "SMAPE": float(smape),
    "MASE": float(mase),
    "RMSSE": float(rmsse),
    "R2": float(r2),
    "NMAE": float(nmae),
    "NRMSE": float(nrmse),
    "STD_ERRORS": float(std_errors)
  }