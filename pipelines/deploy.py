from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import Output
from pydantic import BaseModel
from typing import Annotated
from steps.add_window_features import AddWindowFeatures
from steps.add_lag_features import AddLagFeatures
from steps.add_temporal_features import AddTemporalFeatures
from steps.clean import clean_data
from steps.ingest import ingest_data
from steps.scale import scale_data
from steps.load import load_features
from steps.train import train_model
from steps.split import split_data
from steps.evaluate import evaluate_model
from steps.deployment_trigger import trigger_deployment, DeploymentTrigger
import logging
import config

container_settings = DockerSettings(required_integrations=[MLFLOW])
@pipeline(enable_cache=False,  settings={'docker':container_settings})
def continuous_deployment(
    min_accuracy: Annotated[float, 'min_accuracy'] = 0.92, 
    workers: Annotated[int, 'workers'] = 1, 
    timeout: Annotated[int, 'timeout'] = DEFAULT_SERVICE_START_STOP_TIMEOUT
    ) -> None:
    try:
        # Feature/ETL
        data = ingest_data(DATA_SOURCE=config.DATA_SOURCE)
        data = clean_data(data)
        data = AddTemporalFeatures(data)
        data = AddLagFeatures(data)
        data = AddWindowFeatures(data)
        data = scale_data(data)

        # Train
        X_train, X_test, y_train, y_test = split_data(data)
        model = train_model(X_train, y_train,model_name=f'{config.MODEL_NAME}-XGBoost')
        r2, mape = evaluate_model(model, X_test, y_test)
    
        # Trigger Deployment
        # deployment_decision = trigger_deployment(r2,DeploymentTrigger)
        mlflow_model_deployer_step(
            model=model,
            deploy_decision=True,
            experiment_name=None,
            run_name=None,
            model_name=f'{config.MODEL_NAME}-XGBoost',
            workers= workers,
            mlserver=False,
            timeout=timeout,
        )

    except Exception as e:
        logging.error(e)
        logging.error("Error in continuous deployment")
        logging.error("Stopping the pipeline")
        raise e