import mlflow.sklearn
import mlflow
import xgboost as xgb
import logging
import config
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from numpy import ndarray
from pandas import DataFrame, Series
from zenml import step, client
from typing import Union, Dict
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
logger = logging.getLogger(__name__)

tracker = client.Client().active_stack.experiment_tracker
@step(name='evaluate', experiment_tracker=tracker.name)
def evaluate(data: Dict, model: BaseEstimator, label='TEST') -> bool:
    """
    This step evaluates the model.

    Args:
        data (Union[pd.DataFrame, None]): The input data.

    Returns:
        bool: True if the model is evaluated successfully, False otherwise.
    """
    try:
        logger.info(f'==> Processing evaluate() on {label}')
        split = label.lower()
        X = data[f'X_{split}']
        y = data[f'y_{split}']

        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        r2 = r2_score(y, y_pred) * 100
        mlflow.log_metric(f"mse_{label}", mse)
        mlflow.log_metric(f"mae_{label}", mae)
        mlflow.log_metric(f"mape_{label}", mape)
        mlflow.log_metric(f"r2_{label}", r2)

        return True
    except Exception as e:
        logger.error(f'in evaluate(): {e}')
        return False
