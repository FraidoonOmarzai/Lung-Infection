from src.exception import CustomException
from src.logger import logging
from src.cloud_storage.s3_operations import S3Operation
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact

import sys


class ModelPusher:
    """ The ModelPusher class facilitates the process of pushing machine learning models to a cloud storage service.

    Attributes:
        - model_pusher_config (ModelPusherConfig): An instance of ModelPusherConfig containing configuration parameters for the model pusher.
        - s3_operation (S3Operation): An instance of S3Operation responsible for interacting with Amazon S3.
    """

    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config
        self.s3_operation = S3Operation()

    def init_model_pusher(self):
        try:
            self.s3_operation.sync_folder_to_s3(self.model_pusher_config.LOG_PROD_MODEL_DIR,
                                                self.model_pusher_config.BUCKET_NAME,
                                                self.model_pusher_config.BEST_MODEL_DIR)

            model_pusher_artifact = ModelPusherArtifact(
                self.model_pusher_config.BUCKET_NAME
            )

            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys)
