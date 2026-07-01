from darts.utils.utils import ModelMode, SeasonalityMode

def suggest_exponential_smoothing(trial, prefix=""):
  # Use strings to avoid Optuna warnings, map to enums later
  # Removed 'multiplicative' to prevent crashes on datasets with zero values (endog must be strictly positive)
  trend_str = trial.suggest_categorical(f"{prefix}trend", ["additive", "none"])
  seasonal_str = trial.suggest_categorical(f"{prefix}seasonal", ["additive", "none"])

  # Only allow damped=True if trend is not None
  if trend_str == "none":
    damped = False
  else:
    damped = trial.suggest_categorical(f"{prefix}damped", [True, False])

  if seasonal_str != "none":
    seasonal_periods = trial.suggest_int(f"{prefix}seasonal_periods", 2, 48)
  else:
    seasonal_periods = None

  trend_enum = ModelMode.ADDITIVE if trend_str == "additive" else (ModelMode.MULTIPLICATIVE if trend_str == "multiplicative" else None)
  seasonal_enum = SeasonalityMode.ADDITIVE if seasonal_str == "additive" else (SeasonalityMode.MULTIPLICATIVE if seasonal_str == "multiplicative" else None)

  return {
    "trend": trend_enum,
    "seasonal": seasonal_enum,
    "seasonal_periods": seasonal_periods,
    "damped": damped
  }

def suggest_prophet(trial, prefix=""):
  daily_seasonality = trial.suggest_categorical(f"{prefix}daily_seasonality", [True, False, 'auto'])
  weekly_seasonality = trial.suggest_categorical(f"{prefix}weekly_seasonality", [True, False, 'auto'])
  yearly_seasonality = trial.suggest_categorical(f"{prefix}yearly_seasonality", [True, False, 'auto'])
  interval_width = trial.suggest_float(f"{prefix}interval_width", 0.8, 0.95)

  return {
    "daily_seasonality": daily_seasonality,
    "weekly_seasonality": weekly_seasonality,
    "yearly_seasonality": yearly_seasonality,
    "interval_width": interval_width
  }

def suggest_fft(trial, prefix=""):
  # Removed 'exp' trend to prevent np.log(0) crashes (RuntimeWarning: divide by zero) when datasets contain zero values (common in RAM/CPU metrics at night).
  trend = trial.suggest_categorical(f"{prefix}trend", ['poly', None])
  return {
    "trend": trend
  }


def suggest_transformer(trial, prefix=""):
  input_chunk_length = trial.suggest_int(f"{prefix}input_chunk_length", 12, 48, step=12)
  output_chunk_length = trial.suggest_int(f"{prefix}output_chunk_length", 1, 12)
  d_model = trial.suggest_categorical(f"{prefix}d_model", [16, 32, 64])
  nhead = trial.suggest_categorical(f"{prefix}nhead", [2, 4, 8])
  num_encoder_layers = trial.suggest_int(f"{prefix}num_encoder_layers", 2, 4)
  num_decoder_layers = trial.suggest_int(f"{prefix}num_decoder_layers", 2, 4)
  dropout = trial.suggest_float(f"{prefix}dropout", 0.0, 0.3)
  n_epochs = trial.suggest_int(f"{prefix}n_epochs", 10, 30, step=10)

  return {
    "input_chunk_length": input_chunk_length,
    "output_chunk_length": output_chunk_length,
    "d_model": d_model,
    "nhead": nhead,
    "num_encoder_layers": num_encoder_layers,
    "num_decoder_layers": num_decoder_layers,
    "dropout": dropout,
    "n_epochs": n_epochs,
    "pl_trainer_kwargs": {"accelerator": "auto"}
  }

def suggest_gru(trial, prefix=""):
  input_chunk_length = trial.suggest_int(f"{prefix}input_chunk_length", 12, 48, step=12)
  hidden_dim = trial.suggest_categorical(f"{prefix}hidden_dim", [16, 25, 32])
  n_rnn_layers = trial.suggest_int(f"{prefix}n_rnn_layers", 1, 3)
  dropout = trial.suggest_float(f"{prefix}dropout", 0.0, 0.3)
  n_epochs = trial.suggest_int(f"{prefix}n_epochs", 10, 30, step=10)

  return {
    "input_chunk_length": input_chunk_length,
    "hidden_dim": hidden_dim,
    "n_rnn_layers": n_rnn_layers,
    "dropout": dropout,
    "n_epochs": n_epochs,
    "pl_trainer_kwargs": {"accelerator": "auto"}
  }

def suggest_blockrnngru(trial, prefix=""):
  input_chunk_length = trial.suggest_int(f"{prefix}input_chunk_length", 12, 48, step=12)
  output_chunk_length = trial.suggest_int(f"{prefix}output_chunk_length", 1, 12)
  hidden_dim = trial.suggest_categorical(f"{prefix}hidden_dim", [16, 25, 32])
  n_rnn_layers = trial.suggest_int(f"{prefix}n_rnn_layers", 1, 3)
  dropout = trial.suggest_float(f"{prefix}dropout", 0.0, 0.3)
  n_epochs = trial.suggest_int(f"{prefix}n_epochs", 10, 30, step=10)

  return {
    "input_chunk_length": input_chunk_length,
    "output_chunk_length": output_chunk_length,
    "hidden_dim": hidden_dim,
    "n_rnn_layers": n_rnn_layers,
    "dropout": dropout,
    "n_epochs": n_epochs,
    "pl_trainer_kwargs": {"accelerator": "auto"}
  }

def suggest_ensemble(trial, prefix=""):
  backward_window = trial.suggest_int(f"{prefix}backward_window", 6, 48)
  return {
    "backward_window": backward_window
  }

TUNING_REGISTRY = {
  "ExponentialSmoothing": suggest_exponential_smoothing,
  "Prophet": suggest_prophet,
  "FFT": suggest_fft,
  "Transformer": suggest_transformer,
  "GRU": suggest_gru,
  "BlockRNNGRU": suggest_blockrnngru,
  "Ensemble": suggest_ensemble
}

def suggest_hyperparameters(model_name, trial, prefix=""):
  if model_name not in TUNING_REGISTRY:
    raise ValueError(f"No tuning logic registered for model: {model_name}")
  return TUNING_REGISTRY[model_name](trial, prefix)