from functools import reduce
from operator import mul

import tensorflow as tf


class Pd(tf.keras.layers.Layer):

    def __init__(self, out_shape):
        self._out_shape = out_shape
        self._out_dim = reduce(mul, out_shape)
        self.bias = self.add_variable('bias', self._out_dim)

    def build(self, inputs):
        self.kernel = self.add_variable('kernel', (inputs[-1], self._out_dim),
                                        dtype=tf.float32, initializer=tf.keras.initializers.glorot_uniform())

    def call(self, inputs):
        logits = tf.matmul(self.kernel, inputs) + self.bias
        logits = tf.reshape(logits, (-1,) + self._out_shape)
        return logits


class CategoricalPd(Pd):

    def call(self, inputs, sample=False, greedy=False):
        logits = super().call(inputs)
        if sample:
            return tf.squeeze(tf.argmax(logits, -1)) if greedy else tf.squeeze(tf.multinomial(logits, -1))
        else:
            return logits


class DiagGaussianPd(tf.keras.layers.Layer):

    def __init__(self, out_shape, initial):
        super().__init__(out_shape)
        self._logstd = self.add_variable('logstd', out_shape, dtype=tf.float32,
                                         initializer=tf.constant_initializer(initial))

    def call(self, inputs, sample=False):
        mean = super().call(inputs)
        epsilon = tf.random_normal(self.output_shape)
        return mean + epsilon * self.std

    @property
    def logstd(self):
        return self._logstd

    @property
    def std(self):
        return tf.exp(self._logstd)
