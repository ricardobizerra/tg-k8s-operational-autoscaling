import pandas as pd
from darts import TimeSeries
from darts.models import FFT as _FFT
from models.BaseModel import BaseModel

class FFT(BaseModel):
  def __init__(self, trend=None):
    super().__init__("fft")
    self.model = _FFT(trend=trend) if trend else _FFT()

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
