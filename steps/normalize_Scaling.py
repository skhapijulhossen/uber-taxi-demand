import logging
import pandas as pd

from zenml import step
from typing import Union
from dask import dataframe as dd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def NormalizeScaling(
    data: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame, None]:
    """Normalize scaling step.

    Args:
        data: Input data.
        dd.DataFrame: Input data.

    Returns:
        Normalized data.
    """
    try:
        logger.info(f"==> Processing NormalizeScaling()")
        scaler = StandardScaler()
        scaler.fit(data.drop(columns=['taxi_demand',]))
        data.loc[:, data.columns[:-1]
                ] = scaler.transform(data.drop(columns=['taxi_demand',]))
        logger.info(f'==> Successfully processed NormalizeScaling()')
    except Exception as e:
        logger.error(f"in NormalizeScaling(): {e}")
        return None