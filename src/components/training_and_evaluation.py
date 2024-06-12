from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import ModelTrainingArtifacts
from src.model.model import ModelArchitecture

import mlflow
from urllib.parse import urlparse
import tensorflow as tf
from pathlib import Path
import os
import sys


class ModelTraining:
    def __init__(self, model_training_config: ModelTrainingConfig):
        self.model_training_config = model_training_config

    def prepare_data(self):
        try:

            train_dir = os.path.join(
                self.model_training_config.UNZIP_PATH, 'train')
            test_dir = os.path.join(
                self.model_training_config.UNZIP_PATH, 'val')
            # Setup data inputs
            IMG_SIZE = (224, 224)
            train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                             label_mode="binary",
                                                                             image_size=IMG_SIZE)

            test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                            label_mode="binary",
                                                                            image_size=IMG_SIZE,
                                                                            shuffle=False)  # don't shuffle test data for prediction analysis

            # using tf.data api
            train_data = train_data.cache().shuffle(
                1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            test_data = test_data.cache().shuffle(
                1000).prefetch(buffer_size=tf.data.AUTOTUNE)

            return train_data, test_data
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logging.info("model saved")

    def init_model_training(self):
        try:

            mlflow.tensorflow.autolog()
            logging.info("loading training and validation data...")
            train_data, test_data = self.prepare_data()

            logging.info("loading EfficientNet..._model")
            # model = ModelArchitecture().EfficientNetB0_model()
            model = ModelArchitecture().EfficientNetV2M_model()
            

            logging.info("model compile...")
            model.compile(loss="binary_crossentropy",
                          optimizer=tf.keras.optimizers.Adam(0.001),
                          metrics=["accuracy"])

            mlflow.set_tracking_uri('http://localhost:2024')
            
            mlflow.set_experiment('Tensorflow Models')

            with mlflow.start_run(run_name="Lung Infection Experiments") as mlops_run:

                logging.info("model fit...")
                model.fit(train_data,
                          batch_size=32,
                          epochs=10,
                          validation_data=test_data)

                # evaluate the model
                results = model.evaluate(test_data)
                logging.info(f"Model evaluation: {results}")

                with mlflow.start_run(nested=True):
                    mlflow.log_param('batch size', 32)
                with mlflow.start_run(nested=True):
                    mlflow.log_param('epochs', 10)

                mlflow.log_metric('Evaluation Accuracy', results[1])

                tracking_url_type_store = urlparse(
                    mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.tensorflow.log_model(
                        model, "model", registered_model_name="EfficientNetV2M")
                else:
                    mlflow.tensorflow.log_model(model, "model")

            # logging.info("model evaluation....")
            # results_model = model.evaluate(test_data)
            # logging.info(
            #     f"Model_loss: {results_model[0]}, Model_Accuracy{results_model[1]}")

            # os.makedirs(
            #     self.model_training_config.MODEL_TRAINING_DIR, exist_ok=True)
            # self.save_model(self.model_training_config.SAVE_MODEL_PATH, model)

            # model_training_artifacts = ModelTrainingArtifacts(
            #     self.model_training_config.SAVE_MODEL_PATH)
            # return model_training_artifacts
        except Exception as e:
            raise CustomException(e, sys)
