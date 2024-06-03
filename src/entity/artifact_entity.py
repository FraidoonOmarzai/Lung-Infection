from dataclasses import dataclass 


@dataclass
class DataIngestionArtifacts:
    RSNA_DATA_PATH: str
    
    
@dataclass
class ModelTrainingArtifacts:
    MODEL_TRAINED_PATH: str