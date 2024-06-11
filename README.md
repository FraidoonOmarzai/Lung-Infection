<h1 align=center> Lung-Infection (End-to-end-DL-Project) </h1>

1. git clone the repository

2. define template of the project

3. define setup.py file

4. install the new env and requirements.txt

```bash
conda create -n lung-env python=3.10 -y
conda activate lung-env
pip install -r requirements.txt
```

5. define the logger

6. define the custom exception handler

7. **experiments in google colabs**
    - download the dataset to google colabs
    - EDA the dataset
    - perform feature engineering
        - converting dicom file to jpeg format
        - ready the dataset for training
        - create sample data of 400 for training and 200 for validating
        - download the sample data and store it in S3
    - train the model on the entire dataset
    - using transfer learning and fine tuning
    - model evaluation
    - `note` we used the notebook in google colabs for experiment purposes and creating sample data for training

```bash
## Workflows
- update the constants
- updata the artifact_entity
- update the config_entity
- update the components
- update the pipeline
- update the dvc.yaml
```

8. **Data Ingestion**
    - add s3_operations.py for uploading and downloading files from s3 bucket 
    - we need to run the below command which allows to configure AWS credentials and settings
    ```bash
    aws configure 
    ```
    - follow the above workflows to get the data from the S3 bucket

```bash
# for as many commits you want to undo
git reset HEAD~
```

9. **Model Training and Evaluation**
    - train the model
    - evaluate the model
    - using mlflow 
    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow.db  --host 0.0.0.0 -p 1234 --default-artifact-root ./mlflow_artifacts
    ```