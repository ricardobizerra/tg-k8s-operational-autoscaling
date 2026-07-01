class BaseModel:
  def __init__(self, name):
    self.name = name
    self.model = None

  def fit(self, train, column_name, freq=None, fine_tune=False):
    """
    Fits the model on the training data.
    
    :param train: pandas DataFrame containing the training data
    :param column_name: the name of the column to forecast
    :param freq: the frequency of the time series
    """
    pass

  def predict(self, test, column_name, freq=None):
    """
    Predicts future values based on the test data shape.
    
    :param test: pandas DataFrame containing the test dates
    :param column_name: the name of the column being forecasted
    :param freq: the frequency of the time series
    :return: pandas Series with predictions
    """
    pass