from src.exception import CustomException
from src.logger import logging
from src.constants import *

import tensorflow as tf
import sys


class ModelArchitecture:
    def __init__(self):
        pass

    def EfficientNetB0_model(self):
        try:
            logging.info("model augmentations layer")
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomHeight(0.2),
                tf.keras.layers.RandomWidth(0.2),
                tf.keras.layers.Rescaling(1./255)
            ], name="data_augmentation")
            
            logging.info("Model architecture...")
            # Setup base model and freeze its layers (this will extract features)
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False)
            base_model.trainable = False

            # Setup model architecture with trainable top layers
            inputs = tf.keras.layers.Input(
                shape=(224, 224, 3), name="input_layer")  # shape of input image
            
             # Add in data augmentation Sequential model as a layer
            x = data_augmentation(inputs)
            
            # put the base model in inference mode so we can use it to extract features without updating the weights
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(
                x)  # pool the outputs of the base model
            # x = tf.keras.layers.Dense(256, activation='relu')(x)
            # x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")(
                x)  # same number of outputs as classes
            model = tf.keras.Model(inputs, outputs)

            return model
        except Exception as e:
            raise CustomException(e, sys)

    def EfficientNetV2M_model(self):
        try:
            logging.info("Model architecture...")
            # Setup base model and freeze its layers (this will extract features)
            base_model = tf.keras.applications.EfficientNetV2M(
                include_top=False)
            base_model.trainable = False

            # Setup model architecture with trainable top layers
            inputs = tf.keras.layers.Input(
                shape=(224, 224, 3), name="input_layer")  # shape of input image
            # put the base model in inference mode so we can use it to extract features without updating the weights
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(
                x)  # pool the outputs of the base model
            # x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")(
                x)  # same number of outputs as classes
            model = tf.keras.Model(inputs, outputs)

            return model
        except Exception as e:
            raise CustomException(e, sys)
