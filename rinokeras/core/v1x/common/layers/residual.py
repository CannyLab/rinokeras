"""
Residual Layers
"""
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=F0401
from tensorflow.keras.layers import Layer, Dropout, Dense  # pylint: disable=F0401


class Residual(Model):
    """
    Adds a residual connection between layers. If input to layer is a tuple, adds output to the first element
    of the tuple.
    """
    def __init__(self, layer: Layer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer = layer

    def call(self, inputs, *args, **kwargs):
        layer_out = self.layer(inputs, *args, **kwargs)
        residual = inputs + layer_out

        return residual

    def get_config(self) -> Dict:
        config = {
            'layer': self.layer.__class__.from_config(self.layer.get_config())
        }

        return config


class Highway(Model):
    """
    Implementation of a highway layer. Can use convolutional or fully connected layer.

    From the paper: https://arxiv.org/abs/1607.06450
    """
    def __init__(self,
                 layer,
                 activation: str = 'relu',
                 gate_bias: float = -3.0,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.activation = activation
        self.gate_bias = gate_bias
        self._gate_initializer = tf.keras.initializers.Constant(gate_bias)
        self.dropout = Dropout(0 if dropout is None else dropout)

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.layer = layer

    def build(self, input_shape):
        units = input_shape[-1]
        self.gate = Dense(units=units,
                          activation='sigmoid',
                          use_bias=True,
                          bias_initializer=self._gate_initializer,
                          kernel_regularizer=self.kernel_regularizer,
                          bias_regularizer=self.bias_regularizer,
                          activity_regularizer=self.activity_regularizer)

    def call(self, inputs):
        gated = self.gate(inputs)
        transformed = self.layer(inputs)
        if self.dropout:
            transformed = self.dropout(transformed)
        return gated * transformed + (1 - gated) * inputs

    def get_config(self) -> Dict:
        config = {
            'layer': self.layer.__class__.from_config(self.layer.get_config()),
            'activation': self.activation,
            'gate_bias': self.gate_bias,
            'dropout': self.dropout,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer)
        }

        return config
