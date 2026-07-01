import pandas as pd
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from models.BaseModel import BaseModel
import concurrent.futures

class Ensemble(BaseModel):
  def __init__(self, models, backward_window, forward_window, online_mode=False):
    """
    Initializes the dynamic ensemble model.
    :param models: List of initialized model objects
    :param backward_window: Number of past records to evaluate models against
    :param forward_window: Number of future records to predict at a time
    """
    if not models:
      raise ValueError("Ensemble requires at least one model in the models list.")

    super().__init__("ensemble")
    self.models = {model.name: model for model in models}
    self.backward_window = backward_window
    self.forward_window = forward_window

    self.train_data = None
    self.cumulative_fit_time = 0.0
    self.cumulative_predict_time = 0.0
    self.evaluations_count = 0
    self.last_selected_model_name: str = list(models)[0].name if models else ""

    nn_names = {"transformer", "blockrnngru", "gru"}
    self.is_nn_ensemble = any(name in nn_names for name in self.models.keys())

    self.online_mode = online_mode
    if self.online_mode:
      import random
      try:
        from constants import RANDOM_SEED
      except ImportError:
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from constants import RANDOM_SEED
      
      _rng = random.Random(RANDOM_SEED)
      self._online_best_model_name = _rng.choice(list(self.models.keys()))

  def fit(self, train, column_name, freq=None, fine_tune=False):
    self.train_data = train.copy(deep=True)

    if self.online_mode:
      if len(self.train_data) > self.backward_window:
        eval_train = self.train_data.iloc[:-self.backward_window].reset_index(drop=True)
        actual_result_of_last_record = self.train_data.iloc[-self.backward_window:].reset_index(drop=True)
        fine_tune_kwarg = {"fine_tune": True} if self.is_nn_ensemble else {}

        current_best = self.models[self._online_best_model_name]
        current_best.fit(eval_train, column_name, freq, **fine_tune_kwarg)
        proposed_prediction = current_best.predict(actual_result_of_last_record, column_name, freq)
        proposed_prediction = proposed_prediction.replace([np.inf, -np.inf], np.nan).fillna(value=99999999)

        best_rmse = mean_squared_error(
          actual_result_of_last_record[column_name], 
          proposed_prediction, 
          squared=False
        )

        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
          for key, model in self.models.items():
            if key == self._online_best_model_name:
              continue
            future = executor.submit(
              self._evaluate_model, key, model, eval_train, actual_result_of_last_record, column_name, freq, fine_tune_kwarg
            )
            futures[future] = key

          for future in concurrent.futures.as_completed(futures):
            key, current_rmse, fit_time, pred_time = future.result()
            if current_rmse < best_rmse:
              best_rmse = current_rmse
              self._online_best_model_name = key

      best_model = self.models[self._online_best_model_name]
      fine_tune_kwarg = {"fine_tune": fine_tune} if self.is_nn_ensemble else {}
      best_model.fit(self.train_data, column_name, freq, **fine_tune_kwarg)

  def _evaluate_model(self, key, model, eval_train, actual_result_of_last_record, column_name, freq, fine_tune_kwarg):
    start_fit = time.time()
    model.fit(eval_train, column_name, freq, **fine_tune_kwarg)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    proposed_prediction = model.predict(actual_result_of_last_record, column_name, freq)
    proposed_prediction = proposed_prediction.replace([np.inf, -np.inf], np.nan).fillna(value=99999999)
    pred_time = time.time() - start_pred

    current_rmse = mean_squared_error(
      actual_result_of_last_record[column_name], 
      proposed_prediction, 
      squared=False
    )
    return key, current_rmse, fit_time, pred_time

  def predict(self, test, column_name, freq=None):
    if self.train_data is None:
      raise ValueError("Model must be fitted before predicting.")

    if self.online_mode:
      predicted = self.models[self._online_best_model_name].predict(test, column_name, freq)
      predicted = predicted.replace([np.inf, -np.inf], np.nan).fillna(value=99999999)
      return predicted

    current_train = self.train_data.copy(deep=True)
    backward_window = self.backward_window
    forward_window = self.forward_window

    best_predictions = pd.DataFrame()
    
    import random
    try:
      from constants import RANDOM_SEED
    except ImportError:
      import sys, os
      sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
      from constants import RANDOM_SEED
    
    _rng = random.Random(RANDOM_SEED)
    current_best_model_name = _rng.choice(list(self.models.keys()))

    step_end = len(test)

    for idx in range(0, step_end, forward_window):
      block_start_time = time.time()
      
      current_test_records = test[idx:min(len(test), idx + forward_window)].reset_index(drop=True)
      fine_tune_kwarg = {"fine_tune": (idx > 0)} if self.is_nn_ensemble else {}

      if idx == 0:
        model = self.models[current_best_model_name]

        model.fit(current_train, column_name, freq, **fine_tune_kwarg)
        predicted = model.predict(current_test_records, column_name, freq)
        predicted = predicted.replace([np.inf, -np.inf], np.nan).fillna(value=99999999)

        self.evaluations_count += 1
      else:
        previous_predicted_record_count = min(len(best_predictions), backward_window)

        actual_result_of_last_record = current_train[-previous_predicted_record_count:].reset_index(drop=True)
        previous_prediction = best_predictions[-previous_predicted_record_count:].reset_index(drop=True)

        eval_train = current_train[:-previous_predicted_record_count].reset_index(drop=True)

        best_rmse = float('inf')

        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
          for key, model in self.models.items():
            future = executor.submit(
              self._evaluate_model, key, model, eval_train, actual_result_of_last_record, column_name, freq, fine_tune_kwarg
            )
            futures[future] = key

          for future in concurrent.futures.as_completed(futures):
            key, current_rmse, fit_time, pred_time = future.result()
            self.evaluations_count += 1
            if current_rmse < best_rmse:
              best_rmse = current_rmse
              current_best_model_name = key

        self.last_selected_model_name = current_best_model_name

        best_model = self.models[current_best_model_name]

        best_model.fit(current_train, column_name, freq, **fine_tune_kwarg)
        predicted = best_model.predict(current_test_records, column_name, freq)
        predicted = predicted.replace([np.inf, -np.inf], np.nan).fillna(value=99999999)

        self.evaluations_count += 1

      block_elapsed_time = time.time() - block_start_time
      self.cumulative_fit_time += (block_elapsed_time / 2.0)
      self.cumulative_predict_time += (block_elapsed_time / 2.0)

      timestamp_for_predicted = current_test_records[test.columns[0]]
      best_predictions = pd.concat(
        [
          best_predictions,
          pd.concat([
            timestamp_for_predicted.reset_index(drop=True),
            predicted.reset_index(drop=True)
          ], axis=1)
        ],
        ignore_index=True
      )

      current_train = pd.concat([current_train, current_test_records], ignore_index=True)

    y_pred = best_predictions.iloc[:, -1:].squeeze().reset_index(drop=True)
    y_pred.name = f"{column_name}_Predicted"
    return y_pred
