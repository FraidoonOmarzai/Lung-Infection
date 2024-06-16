from src.exception import CustomException
from src.logger import logging
from src.components.log_production_model import LogProductionModel
from src.entity.artifact_entity import LogProductionModelArtifacts
from src.entity.config_entity import LogProductionModelConfig

import sys


class LogProductionModelPipeline:
    """Class representing a log production model pipeline.

    Methods:
        start_model_training: Starts the model_training process.
        run_pipeline: Runs the training pipeline.
    """

    def __init__(self):
        self.log_prod_model_config = LogProductionModelConfig()

    def start_log_prod_model(self):
        try:
            logging.info("Starting model training pipeline...")
            log_production_model = LogProductionModel(
                self.log_prod_model_config)
            log_production_model_artificats = log_production_model.init_log_production_model()
            return log_production_model_artificats

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):

        logging.info("Running log prod pipeline...")
        log_production_model_artificats: LogProductionModelArtifacts = self.start_log_prod_model()

        logging.info("Log Production Model pipeline done...")


if __name__ == '__main__':
    try:
        log_prod_pipeline = LogProductionModelPipeline()
        log_prod_pipeline.run_pipeline()
    except Exception as e:
        raise CustomException(e, sys)
