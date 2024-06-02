from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts
from src.cloud_storage.s3_operations import S3Operation

import zipfile
import sys
import os


class DataIngestion:
    """Class for ingesting data from S3 and performing necessary operations.

    This class facilitates the process of downloading data from an S3 bucket,
    unzipping the downloaded data, and creating a data ingestion artifact.


    Methods:
        get_data_from_s3: Downloads data from S3 to the local system.
        unzip_data: Unzips the downloaded data.
        init_data_ingestion: Initializes the data ingestion process.
    """

    def __init__(self,
                 data_ingestion_config: DataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config
        self.s3_operation = S3Operation()

    def get_data_from_s3(self):
        try:
            logging.info("calling s3_operation.get_data_from_s3...")

            os.makedirs(
                self.data_ingestion_config.DATA_INDESTION_PATH, exist_ok=True)
            self.s3_operation.sync_folder_from_s3(self.data_ingestion_config.DATA_INDESTION_PATH,
                                                  self.data_ingestion_config.BUCKET_NAME,
                                                  self.data_ingestion_config.S3_DATA_FOLDER)
            logging.info("Downloading completed successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def unzip_data(self):
        try:
            logging.info("unzipping data starting...")
            with zipfile.ZipFile(self.data_ingestion_config.ZIP_PATH, 'r') as zipped_file:
                zipped_file.extractall(self.data_ingestion_config.UNZIP_PATH)

        except Exception as e:
            raise CustomException(e, sys)

    def init_data_ingestion(self):
        try:
            if not os.path.exists(self.data_ingestion_config.UNZIP_PATH):
                self.get_data_from_s3()
                self.unzip_data()
            else:
                logging.info("data already exists")

            data_ingestion_artifacts: DataIngestionArtifacts = DataIngestionArtifacts(
                self.data_ingestion_config.UNZIP_PATH
            )

            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
