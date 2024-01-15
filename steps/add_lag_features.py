from zenml import step
from dask import dataframe as dd
import logging
import pandas as pd
from typing import Union
from feature_engine.timeseries.forecasting import LagFeatures

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def AddLagFeatures(data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """
    Add lag features to the data.
    """
    logger.info(f"Adding lag features to the data.")
    try:
        lagfeatures = LagFeatures(variables=None, periods=[1, 2, 4, 8, 16, 24], freq=None, sort_index=True,
                                  missing_values='raise', drop_original=False)
        lagfeatures.fit(data[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = lagfeatures.transform(
            data[['timestamp', 'passenger_demand', 'taxi_demand']])
        for col in list(features.columns)[3:]:
            data[col] = features[col].values
        logger.info(f'==> Successfully processed add_lag_features()')
        return data
    except Exception as e:
        logger.error(f'in The add_lag_features(): {e}')
        return None
