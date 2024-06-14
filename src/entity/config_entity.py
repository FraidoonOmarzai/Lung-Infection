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
    def __init__(self) -> None:
        self.MODEL_TRAINING_DIR = os.path.join(MODEL_TRAINING_DIR)
        self.UNZIP_PATH = os.path.join(UNZIP_DIR)
        self.SAVE_MODEL_PATH = os.path.join(self.MODEL_TRAINING_DIR, SAVE_MODEL_NAME)
        
        

class LogProductionModelConfig:
    def __init__(self) -> None:
        self.LOG_PRO_MODEL_DIR = os.path.join(LOG_PRO_MODEL_DIR)
        