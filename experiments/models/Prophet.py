import pandas as pd
from darts import TimeSeries
from darts.models import Prophet as _Prophet
from models.BaseModel import BaseModel

class Prophet(BaseModel):
  def __init__(self, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, interval_width=0.95):
    super().__init__("prophet")
    self.model = _Prophet(
      daily_seasonality=daily_seasonality,
      weekly_seasonality=weekly_seasonality,
      yearly_seasonality=yearly_seasonality,
      interval_width=interval_width
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

    # convert to pandas series
    y_pred = forecast.pd_series().reset_index(drop=True)
    y_pred.name = f"{column_name}_Predicted"
    
    return y_pred
