import sys
import os

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, 'steps'))


# Now you can import modules from the parent directory
from zenml import pipeline
from steps.add_window_features import AddWindowFeatures
from steps.add_lag_features import AddLagFeatures
from steps.add_temporal_features import AddTemporalFeatures
from steps.clean import clean_data
from steps.ingest import ingest_data
from steps.scale import scale_data
from steps.load import load_features
import logging
import config

@pipeline(name='ETLFeaturePipelineUberTaxiDemand', enable_step_logs=True)
def run_pipeline():
    """
    Pipeline that runs the ingest, clean, lag and window features.
    """
    try:
        data = ingest_data(DATA_SOURCE=config.DATA_SOURCE)
        data = clean_data(data)
        data = AddTemporalFeatures(data)
        data = AddLagFeatures(data)
        data = AddWindowFeatures(data)
        data = scale_data(data)
        # success = load_features(data)
    except Exception as e:
        logging.error(f'==> Error in run_pipeline(): {e}')


if __name__ == "__main__":
    run = run_pipeline()
