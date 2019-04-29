"""
Normalization style layers
"""

import collections
from typing import Union, Sequence, Dict

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.keras.layers import Layer, Dense  # pylint: disable=F0401

# https://github.com/keras-team/keras/issues/3878
class LayerNorm(Layer):
    """
    Does layer normalization from https://arxiv.org/abs/1607.06450.
    """

    def __init__(self, axis: Union[Sequence[int], int] = -1, eps: float = 1e-6, trainable=True, **kwargs) -> None:
        super().__init__(trainable=trainable, **kwargs)
        #TODO: Proper type hints here. (Sequence[int]) ?
        self.axis = axis if isinstance(axis, collections.Sequence) else (axis,)
        self.eps = eps
        self.trainable = trainable

    def build(self, input_shape):
        # shape = [input_shape[axis] for axis in self.axis]
        shape = input_shape[-1:]

        self.gamma = self.add_variable(name='gamma',
                                       shape=shape,
                                       initializer=tf.keras.initializers.Ones(),
                                       trainable=self.trainable)
        self.beta = self.add_variable(name='beta',
                                      shape=shape,
                                      initializer=tf.keras.initializers.Zeros(),
                                      trainable=self.trainable)
        super().build(input_shape)

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, self.axis, keep_dims=True)
        # mean = K.mean(inputs, axis=self.axis, keepdims=True)
        # std = K.std(inputs, axis=self.axis, keepdims=True)
        # return self.gamma * (inputs - mean) / (std + self.eps) + self.beta
        normalized = tf.nn.batch_normalization(inputs, mean, var, self.beta, self.gamma, self.eps)
        return normalized

    def get_config(self) -> Dict:
        config = {
            'axis': self.axis,
            'eps': self.eps
        }
        return config


class WeightNormDense(Dense):

    def build(self, input_shape):
        super().build(input_shape)
        self.scale = self.add_weight(
            'g',
            [self.units],
            initializer='ones',
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs):
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=0) * self.scale
        return super().call(inputs)
