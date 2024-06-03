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
            logging.info("Model architecture...")
            # Setup base model and freeze its layers (this will extract features)
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False)
            base_model.trainable = False

            # Setup model architecture with trainable top layers
            inputs = tf.keras.layers.Input(
                shape=(224, 224, 3), name="input_layer")  # shape of input image
            # put the base model in inference mode so we can use it to extract features without updating the weights
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(
                x)  # pool the outputs of the base model
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer")(
                x)  # same number of outputs as classes
            model = tf.keras.Model(inputs, outputs)

            return model
        except Exception as e:
            raise CustomException(e, sys)
