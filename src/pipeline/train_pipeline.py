from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.training_and_evaluation import ModelTraining
from src.components.log_production_model import LogProductionModel
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import (DataIngestionConfig,
                                      ModelTrainingConfig,
                                      LogProductionModelConfig,
                                      ModelPusherConfig)
from src.entity.artifact_entity import (DataIngestionArtifacts,
                                        ModelTrainingArtifacts,
                                        LogProductionModelArtifacts,
                                        ModelPusherArtifact)
from src.components.data_ingestion import DataIngestion

import sys


class TrainingPipeline:
    """Class representing a training pipeline.

    This class orchestrates the various steps involved in the training pipeline,
    including data ingestion, model training and further processing.

    Methods:
        start_data_ingestion: Starts the data ingestion process.
        start_model_training: Starts the model training process.
        start log_production_model: Starts the  log_production_model process.
        start_model_pusher: Starts the model pusher process.
        run_pipeline: Runs the entire training pipeline.
    """

    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.model_training_config = ModelTrainingConfig()
        self.log_prod_model_config = LogProductionModelConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion pipeline...")

            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.init_data_ingestion()
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self):
        try:
            logging.info("Starting model training pipeline...")
            model_training = ModelTraining(self.model_training_config)
            model_training_artifacts = model_training.init_model_training()

            return model_training_artifacts

        except Exception as e:
            raise CustomException(e, sys)

    def start_log_prod_model(self):
        try:
            logging.info("Starting model training pipeline...")
            log_production_model = LogProductionModel(
                self.log_prod_model_config)
            log_production_model_artificats = log_production_model.init_log_production_model()
            return log_production_model_artificats

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(self):
        try:
            logging.info("Starting model pusher pipeline...")
            pusher_model = ModelPusher(self.model_pusher_config)
            model_pusher_artifact = pusher_model.init_model_pusher()

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        logging.info("running training pipeline")
        try:
            # data ingestion Sections
            print("<====================Data Ingestions===========================>")
            data_ingestion_artifacts: DataIngestionArtifacts = self.start_data_ingestion()
            logging.info("Data ingestion pipeline done...")

            # Model Training Sections
            print("<====================Model Training===========================>")
            model_training_artifacts: ModelTrainingArtifacts = self.start_model_training()
            logging.info("Model training pipeline done...")

            # Log Production Model Sections
            print("<====================Log Production Model===========================>")
            log_production_model_artificats: LogProductionModelArtifacts = self.start_log_prod_model()
            logging.info("Log Production Model pipeline done...")

            # Model Pushing Sections
            print("<====================Model Pusher===========================>")
            model_pusher_artificats: ModelPusherArtifact = self.start_model_pusher()
            logging.info("model pusher pipeline done...")

        except Exception as e:
            raise CustomException(e, sys)
