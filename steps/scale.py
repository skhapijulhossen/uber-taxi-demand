from zenml import step
import logging
import pandas as pd
from typing import Union
from sklearn.preprocessing import StandardScaler
import joblib
import os
import config

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def scale_data(data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """Scaling step.
    Args:
        data: Input data.
    Returns:
        Normalized data.
    """
    try:
        logger.info(f'==> Processing scale_data()')
        scaler = StandardScaler()
        # Assuming the data is a pandas DataFrame
        temp = data[['timestamp', 'taxi_demand']]
        data.drop(columns=['taxi_demand', 'timestamp'], inplace=True)
        scaler.fit(data)
        data = pd.concat(
            [temp, pd.DataFrame(scaler.transform(data), columns=data.columns)], axis=1)
        del temp
        # save Scaler model
        joblib.dump(scaler, os.path.join('model', 'scaler.pkl'))
        logger.info(f'Scaler model saved to {os.path.join("model", "scaler.pkl")}')
        print(data.columns)
        logger.info(f'==> Successfully processed scale_data()')
        return data
    except Exception as e:
        logger.error(f"in scale_data(): {e}")
        return None


if __name__ == "__main__":
    data = pd.read_csv("data/train.csv")
    print(scale_data(data))
