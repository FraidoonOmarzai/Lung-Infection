from src.exception import CustomException
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
import sys

class StartTraining:
    try:
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run_pipeline()
    except Exception as e:
        raise CustomException(e, sys)
        
if __name__ =="__main__":
    StartTraining()