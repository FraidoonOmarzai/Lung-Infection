from src.constants import *
import os


class DataIngestionConfig:
    """Data class for configuring data ingestion parameters.

    This class provides a convenient way to configure parameters related to data ingestion,
    such as directory paths and S3 bucket names.

    args:

    """

    def __init__(self) -> None:
        self.S3_DATA_FOLDER = S3_DATA_FOLDER
        self.BUCKET_NAME = BUCKET_NAME
        self.S3_FILE = S3_FILE

        self.DATA_INDESTION_PATH = os.path.join(DATA_INGESTION_DIR)

        self.ZIP_PATH = os.path.join(self.DATA_INDESTION_PATH, S3_FILE)
        self.UNZIP_PATH = os.path.join(self.DATA_INDESTION_PATH, IMAGES_DIR)


class ModelTrainingConfig:
    """A class to hold configuration parameters for model training.

    This class provides a convenient way to configure parameters related to model training and evaluation.

    args:

    """

    def __init__(self) -> None:
        self.MODEL_TRAINING_DIR = os.path.join(MODEL_TRAINING_DIR)
        self.UNZIP_PATH = os.path.join(UNZIP_DIR)
        self.TRAIN_DIR = os.path.join(self.UNZIP_PATH, 'train')
        self.VAL_DIR = os.path.join(self.UNZIP_PATH, 'val')
        self.SAVE_MODEL_PATH = os.path.join(
            self.MODEL_TRAINING_DIR, SAVE_MODEL_NAME)

        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS


class LogProductionModelConfig:
    """A class to hold configuration parameters for log production model.
    args:

    """

    def __init__(self) -> None:
        self.LOG_PROD_MODEL_DIR = os.path.join(LOG_PRO_MODEL_DIR)
        self.LOG_SAVE_MODEL_PATH = os.path.join(
            self.LOG_PROD_MODEL_DIR, LOG_SAVE_MODEL_NAME)


class ModelPusherConfig:
    """A class to hold configuration parameters for model pusher.

    Attributes:
        log_prod_model_dir (str): Path to log_prod_model directory.
        bucket_name (str): Name of the cloud storage bucket.
        best_model (str): Directory name for the best model within the cloud storage.

    Methods:
        __init__():
            Initialize ModelPusherConfig with default or provided settings.
    """

    def __init__(self):
        self.LOG_PROD_MODEL_DIR = os.path.join(LOG_PRO_MODEL_DIR)

        self.BUCKET_NAME = BUCKET_NAME
        self.BEST_MODEL_DIR = BEST_MODE_DIR
