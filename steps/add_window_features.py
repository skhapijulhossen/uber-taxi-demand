from zenml import step
from dask import dataframe as dd
import logging
import pandas as pd
from typing import Union
from feature_engine.timeseries.forecasting import WindowFeatures

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def AddWindowFeatures(
        data: Union[dd.DataFrame, pd.DataFrame]) -> Union[dd.DataFrame, pd.DataFrame]:
    """Add window features to the dataframe

    Args:
        data (Union[dd.DataFrame, pd.DataFrame]): The dataframe to add window features to.

    Returns:
        Union[dd.DataFrame, pd.DataFrame]: The dataframe with window features added.
    """
    logger.info("Adding window features to the dataframe")

    try:
        windowfeatures = WindowFeatures(variables=None, window=24, freq=None, sort_index=True,
                                        missing_values='raise', drop_original=False)
        windowfeatures.fit(
            data[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = windowfeatures.transform(
            data[['timestamp', 'passenger_demand', 'taxi_demand']])
        for col in list(features.columns)[3:]:
            data[col] = features[col].values
        logger.info(f'==> Successfully processed add_window_features()')
        return data
    except Exception as e:
        logger.error(f'in add_window_features(): {e}')
        return None
