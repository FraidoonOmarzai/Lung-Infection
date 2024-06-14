from src.exception import CustomException
from src.logger import logging

import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os
import sys

class LogProductionModel:
    def __init__(self):
        pass
    def log_production(self):
        try:
            logging.info("set the tracking uri")
            mlflow.set_tracking_uri('http://localhost:2024')
            
            # get all the experiments details
            runs = mlflow.search_runs(experiment_ids=[1])
            logging.info(f"runs: {runs}")
            
            # get the experiment with highest validation accuracy
            highest = runs["metrics.validation_accuracy"].sort_values(ascending=False, ignore_index=True)[0]
            
            # get the experiment with highest validation accuracy run id
            highest_run_id = runs[runs["metrics.validation_accuracy"]==highest]['run_id']
            highest_run_id = highest_run_id.reset_index(drop=True)[0]
            logging.info(f"highest run id: {highest_run_id}")
            
            
            
            # 
            model_name = 'EfficientNetV2M'

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
            loaded_model = mlflow.pyfunc.load_model(logged_model)

            os.makedirs(os.path.join('artifacts\LogProductionModel'), exist_ok=True)
            model_path = os.path.join('artifacts\LogProductionModel\mlflow_model.h5')
            joblib.dump(loaded_model, model_path)
            logging.info("model saved!")
        except Exception as e:
            raise CustomException(e, sys)
        
    def init_log_production_model(self):
        try:
            self.log_production()
        except Exception as e:
            raise CustomException(e, sys)