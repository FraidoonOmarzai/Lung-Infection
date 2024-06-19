from src.exception import CustomException
from src.logger import logging
from src.components.model_pusher import ModelPusher
from src.entity.artifact_entity import ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig

import sys


class ModelPusherlPipeline:
    """Class representing a model pusher pipeline.

    Methods:
        start_model_pusher: Starts the model_pusher process.
        run_pipeline: Runs the pusher pipeline.
    """

    def __init__(self):
        self.model_pusher_config = ModelPusherConfig()

    def start_model_pusher(self):
        try:
            logging.info("Starting model pusher pipeline...")
            pusher_model = ModelPusher(self.model_pusher_config)
            model_pusher_artifact = pusher_model.init_model_pusher()

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):

        logging.info("Running log prod pipeline...")
        model_pusher_artificats: ModelPusherArtifact = self.start_model_pusher()

        logging.info("Log Production Model pipeline done...")


if __name__ == '__main__':
    try:
        pusher_pipeline = ModelPusherlPipeline()
        pusher_pipeline.run_pipeline()
    except Exception as e:
        raise CustomException(e, sys)
