from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts
from src.components.data_ingestion import DataIngestion

import sys

class DataIngestionPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.init_data_ingestion()
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        data_ingestion_artifacts: DataIngestionArtifacts = self.start_data_ingestion()
        
        
        
if __name__ == '__main__':
    try:
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run_pipeline()
    except Exception as e:
        raise CustomException(e, sys)