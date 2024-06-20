from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import ModelTrainingArtifacts
from src.model.model import ModelArchitecture
from src.constants import *

import mlflow
from urllib.parse import urlparse
import tensorflow as tf
from pathlib import Path
import os
import sys


class ModelTraining:
    """A class for training DL models.

    Attributes:
        - model_training_config (ModelTrainingConfig): Configuration parameters for model training.

    Methods:
        - prepare_data(): Prepare the images for training.
        - save_model(): Save the model locally.
        - init_model_training(): Load the model architecture, use mlflow and traing the model.
    """

    def __init__(self, model_training_config: ModelTrainingConfig):
        self.model_training_config = model_training_config

    def prepare_data(self):
        try:

            train_dir = self.model_training_config.TRAIN_DIR
            test_dir = self.model_training_config.VAL_DIR
            # Setup data inputs
            IMG_SIZE = self.model_training_config.IMAGE_SIZE
            BATCH_SIZE = self.model_training_config.BATCH_SIZE
            train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                             label_mode="binary",
                                                                             batch_size=BATCH_SIZE,
                                                                             image_size=IMG_SIZE)

            test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                            label_mode="binary",
                                                                            batch_size=BATCH_SIZE,
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
            model = ModelArchitecture().EfficientNetB0_model()
            # model = ModelArchitecture().EfficientNetV2M_model()

            logging.info("model compile...")
            model.compile(loss="binary_crossentropy",
                          optimizer=tf.keras.optimizers.Adam(0.001),
                          metrics=["accuracy"])

            mlflow.set_tracking_uri(REMOTE_SERVER_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)

            with mlflow.start_run(run_name=RUN_NAME) as mlops_run:

                logging.info("model fit...")
                model.fit(train_data,
                          epochs=self.model_training_config.EPOCHS,
                          validation_data=test_data)

                # evaluate the model
                results = model.evaluate(test_data)
                logging.info(f"Model evaluation: {results}")

                with mlflow.start_run(nested=True):
                    mlflow.log_param('batch size', BATCH_SIZE)
                with mlflow.start_run(nested=True):
                    mlflow.log_param('epochs', EPOCHS)

                mlflow.log_metric('Evaluation Accuracy', results[1])

                tracking_url_type_store = urlparse(
                    mlflow.get_tracking_uri()).scheme
                if tracking_url_type_store != "file":
                    # Register the model
                    mlflow.tensorflow.log_model(
                        model, "model", registered_model_name=REGISTERED_MODEL_NAME)
                else:
                    mlflow.tensorflow.log_model(model, "model")

            logging.info("Current trained model evaluation....")
            results_model = model.evaluate(test_data)
            logging.info(
                f"Model_loss: {results_model[0]}, Model_Accuracy{results_model[1]}")

            os.makedirs(
                self.model_training_config.MODEL_TRAINING_DIR, exist_ok=True)
            self.save_model(self.model_training_config.SAVE_MODEL_PATH, model)
            logging.info("trained model saved successfully")

            model_training_artifacts = ModelTrainingArtifacts(
                self.model_training_config.SAVE_MODEL_PATH)
            return model_training_artifacts
        except Exception as e:
            raise CustomException(e, sys)
