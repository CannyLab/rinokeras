"""
Activation-style layers 
"""

from typing import Dict
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=F0401
from tensorflow.keras.layers import Dense  # pylint: disable=F0401


class GatedTanh(Model):

    def __init__(self, n_units, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
        super().__init__()

        self.n_units = n_units
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        self.fc = Dense(n_units, 'tanh', kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer)
        self.gate = Dense(n_units, 'sigmoid', kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer)

    def call(self, inputs):
        return self.fc(inputs) * self.gate(inputs)

    def get_config(self) -> Dict:
        config = {
            'n_units': self.n_units,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer)
        }

        return config
