import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing as _ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode
from models.BaseModel import BaseModel

import constants

class ExponentialSmoothing(BaseModel):
  def __init__(self, trend=None, seasonal=None, seasonal_periods=None, damped=False):
    super().__init__("exponential_smoothing")

    # Optuna strings to Darts Enums
    if isinstance(trend, str):
      if trend == "none": trend = None
      elif trend == "additive": trend = ModelMode.ADDITIVE
      elif trend == "multiplicative": trend = ModelMode.MULTIPLICATIVE

    if isinstance(seasonal, str):
      if seasonal == "none": seasonal = None
      elif seasonal == "additive": seasonal = SeasonalityMode.ADDITIVE
      elif seasonal == "multiplicative": seasonal = SeasonalityMode.MULTIPLICATIVE
    
    self.model = _ExponentialSmoothing(
      trend=trend,
      seasonal=seasonal,
      seasonal_periods=seasonal_periods,
      damped=damped,
      random_state=constants.RANDOM_SEED
    )

  def fit(self, train, column_name, freq=None):
    train_series = TimeSeries.from_dataframe(
      train, 
      time_col=train.columns[0], 
      value_cols=column_name,
      fill_missing_dates=True,
      freq=freq 
    )

    self.model.fit(train_series)

  def predict(self, test, column_name, freq=None):
    forecast = self.model.predict(n=len(test))
    
    y_pred = forecast.pd_series().reset_index(drop=True)
    y_pred.name = f"{column_name}_Predicted"
    
    return y_pred
