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

    def __init__(self, axis: Union[Sequence[int], int] = -1, eps: float = 1e-6, **kwargs) -> None:
        super().__init__(**kwargs)
        #TODO: Proper type hints here. (Sequence[int]) ?
        self.axis = axis if isinstance(axis, collections.Sequence) else (axis,)
        self.eps = eps

    def build(self, input_shape):
        # shape = [input_shape[axis] for axis in self.axis]
        shape = input_shape[-1:]

        self.gamma = self.add_variable(name='gamma',
                                       shape=shape,
                                       initializer=tf.keras.initializers.Ones(),
                                       trainable=True)
        self.beta = self.add_variable(name='beta',
                                      shape=shape,
                                      initializer=tf.keras.initializers.Zeros(),
                                      trainable=True)
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
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.kernel)

        scale = self.scale / (tf.norm(self.kernel, 2, 0) + 1e-8)
        outputs = outputs * scale
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
