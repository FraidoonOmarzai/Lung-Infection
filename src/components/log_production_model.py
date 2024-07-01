from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import LogProductionModelArtifacts
from src.entity.config_entity import LogProductionModelConfig
from src.constants import *

import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os
import sys
from pathlib import Path


class LogProductionModel:
    """A class for Log Production models.
    This class facilitates the process of getting the best model from mlflwo, registering models in 
    stagging and production, and save the model.    
    Attributes:
        - log_model_prod_config (LogProductionModelConfig): Configuration parameters.

    Methods:
        - log_production()
        - init_log_production_model()
    """

    def __init__(self, log_model_prod_config: LogProductionModelConfig):
        self.log_model_prod_config = log_model_prod_config

    def log_production(self):
        try:
            logging.info("set the tracking uri")
            mlflow.set_tracking_uri(REMOTE_SERVER_URI)

            # get all the experiments details
            runs = mlflow.search_runs(experiment_ids=[2])
            logging.info(f"runs: {runs}")

            # get the experiment with highest validation accuracy
            highest = runs["metrics.validation_accuracy"].sort_values(
                ascending=False, ignore_index=True)[0]

            # get the experiment with highest validation accuracy run id
            highest_run_id = runs[runs["metrics.validation_accuracy"]
                                  == highest]['run_id']
            highest_run_id = highest_run_id.reset_index(drop=True)[0]
            logging.info(f"highest run id: {highest_run_id}")

            model_name = REGISTERED_MODEL_NAME

            client = MlflowClient()
            for mv in client.search_model_versions(f"name='{model_name}'"):
                mv = dict(mv)

                if mv["run_id"] == highest_run_id:
                    current_version = mv["version"]
                    logged_model = mv["source"]
                    pprint(mv, indent=4)
                    client.transition_model_version_stage(
                        name=model_name,
                        version=current_version,
                        stage="Production"
                    )
                else:
                    current_version = mv["version"]
                    client.transition_model_version_stage(
                        name=model_name,
                        version=current_version,
                        stage="Staging"
                    )

            # load the model with highest accuracy and save it
            # loaded_model = mlflow.pyfunc.load_model(logged_model)
            loaded_model = mlflow.tensorflow.load_model(logged_model)
            

            # stored the best model locally
            os.makedirs(
                self.log_model_prod_config.LOG_PROD_MODEL_DIR, exist_ok=True)
            # joblib.dump(loaded_model,
            #             self.log_model_prod_config.LOG_SAVE_MODEL_PATH)
            # joblib.dump(loaded_model,
            #             "artifacts\LogProductionModel\mlflow_model.h5")
            
            loaded_model.save(Path("artifacts\LogProductionModel\mlflow_model.h5"))
            logging.info("model saved!")

        except Exception as e:
            raise CustomException(e, sys)

    def init_log_production_model(self):
        try:
            self.log_production()

            log_production_model_artificats = LogProductionModelArtifacts(
                self.log_model_prod_config.LOG_SAVE_MODEL_PATH
            )
            return log_production_model_artificats
        except Exception as e:
            raise CustomException(e, sys)
