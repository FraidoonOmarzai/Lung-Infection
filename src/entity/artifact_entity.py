from dataclasses import dataclass 


@dataclass
class DataIngestionArtifacts:
    RSNA_DATA_PATH: str
    
    
@dataclass
class ModelTrainingArtifacts:
    MODEL_TRAINED_PATH: str
    
    
@dataclass
class LogProductionModelArtifacts:
    MODEL_PRODUCTION_PATH: str
    
    
@dataclass
class ModelPusherArtifact:
    bucket_name: str