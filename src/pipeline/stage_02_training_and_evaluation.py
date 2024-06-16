from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import ModelTrainingArtifacts
from src.components.training_and_evaluation import ModelTraining

import sys


class ModelTrainingPipeline:
    """Class representing a model training pipeline.

    Methods:
        start_model_training: Starts the model_training process.
        run_pipeline: Runs the training pipeline.
    """

    def __init__(self):
        self.model_training_config = ModelTrainingConfig()

    def start_model_training(self):
        try:
            logging.info("Starting model training pipeline...")
            model_training = ModelTraining(self.model_training_config)
            model_training_artifacts = model_training.init_model_training()

            return model_training_artifacts

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):

        logging.info("Running training pipeline...")
        model_training_artifacts: ModelTrainingArtifacts = self.start_model_training()

        logging.info("Model training pipeline done...")


if __name__ == '__main__':
    try:
        model_training_pipeline = ModelTrainingPipeline()
        model_training_pipeline.run_pipeline()
    except Exception as e:
        raise CustomException(e, sys)
