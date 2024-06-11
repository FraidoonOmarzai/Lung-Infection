import os


# COMMON CONSTANTS
ARTIFACTS_DIR: str = os.path.join('artifacts')
BUCKET_NAME: str = 'rsnadataset'
S3_DATA_FOLDER: str = 'data'
S3_FILE: str = 'RSNA.zip'


# DATA INGESTION CONSTANTS
DATA_INGESTION_DIR: str = os.path.join(ARTIFACTS_DIR, 'DataIngestion')
IMAGES_DIR: str = "images"


# MODET TRAINING CONSTANS
MODEL_TRAINING_DIR: str = os.path.join(ARTIFACTS_DIR, 'ModelTrainings')
UNZIP_DIR: str = os.path.join(DATA_INGESTION_DIR, IMAGES_DIR)
SAVE_MODEL_NAME: str = 'model.h5'
EPOCHS: int = 5


# mlflow CONSTANTS
artifacts_dir: str = "mlflow_artifacts"
experiment_name: str = "Lung Infection Experiments"
run_name: str = "Transfer_learning_Model"
registered_model_name: str = "EfficientNetB0"
remote_server_uri: str = "http://localhost:1234"