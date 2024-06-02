from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import ModelTrainingArtifacts

import tensorflow as tf
from pathlib import Path
import os
import sys


class ModelTraining:
    def __init__(self, model_training_config: ModelTrainingConfig):
        self.model_training_config = model_training_config
        
    def prepare_data(self):
        try:
            
            train_dir = os.path.join(self.model_training_config.UNZIP_PATH, 'train')
            test_dir = os.path.join(self.model_training_config.UNZIP_PATH, 'val')
            # Setup data inputs
            IMG_SIZE = (224, 224)
            train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                            label_mode="binary",
                                                                            image_size=IMG_SIZE)

            test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                            label_mode="binary",
                                                                            image_size=IMG_SIZE,
                                                                            shuffle=False) # don't shuffle test data for prediction analysis
            
            
            train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            test_data = test_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            
            return train_data, test_data
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def trainings(self):
        try:
            
            # Setup base model and freeze its layers (this will extract features)
            base_model = tf.keras.applications.EfficientNetB0(include_top=False)
            base_model.trainable = False

            # Setup model architecture with trainable top layers
            inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer") # shape of input image
            x = base_model(inputs, training=False) # put the base model in inference mode so we can use it to extract features without updating the weights
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x) # pool the outputs of the base model
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")(x) # same number of outputs as classes
            model = tf.keras.Model(inputs, outputs)
            
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logging.info("model saved")
    
    def init_model_training(self):
        try:
            
            train_data, test_data = self.prepare_data()  
            model = self.trainings()
            
            # Compile
            model.compile(loss="binary_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(0.001),
                        metrics=["accuracy"])
            
            # Fit
            model.fit(train_data,
                    epochs=5,
                    validation_data=test_data
                    )
            
            os.makedirs(self.model_training_config.MODEL_TRAINING_DIR, exist_ok=True)
            self.save_model(self.model_training_config.SAVE_MODEL_PATH, model)
        except Exception as e:
            raise CustomException(e, sys)