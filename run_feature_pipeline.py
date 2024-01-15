from zenml import pipeline
from steps.ingest import ingest_data, optimizeToFitMemory
from steps.clean import clean_data
from steps.add_temporal_features import AddTemporalFeatures
from steps.add_lag_features import AddLagFeatures
from steps.add_window_features import AddWindowFeatures
import config
import logging


@pipeline(enable_cache=True)
def run_pipeline():
    """
    Pipeline that runs the ingest, clean, lag and window features.
    """
    try:
        logging.info(f'==> Processing run_pipeline()')
        data = ingest_data(DATA_SOURCE=config.DATA_SOURCE)
        data = clean_data(data)
        data = AddTemporalFeatures(data)
        data = AddLagFeatures(data)
        data = AddWindowFeatures(data)
        logging.info(f'==> Successfully processed run_pipeline()')
    except Exception as e:
        logging.error(f'==> Error in run_pipeline(): {e}')


if __name__ == "__main__":
    run = run_pipeline()
