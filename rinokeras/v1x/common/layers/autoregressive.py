"""
AutoRegressive layers
#TODO(Roshan): Rename these so that they're actually indicitive of what they do.
"""

from typing import Dict, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=F0401
from tensorflow.keras.layers import Dense  # pylint: disable=F0401


class RandomGaussNoise(tf.keras.layers.Layer):
    """
    Adds gaussian random noise to input with trainable standard deviation.
    """

    def __init__(self, noise_shape: Optional[Tuple[int, ...]] = None, initial_logstd: float = 0, dtype=tf.float32) -> None:
        super().__init__()
        self._noise_shape = noise_shape
        self._initial_logstd = initial_logstd
        self._dtype = dtype

    def build(self, input_shape):
        if self._noise_shape is not None:
            shape = self._noise_shape
            if not input_shape[1:].is_compatible_with(tuple(dim if dim != 1 else None for dim in shape)):
                raise ValueError("Shapes {} and {} are incompatible and cannot be broadcasted".format(
                    input_shape[1:], shape))
        else:
            shape = input_shape[1:]
        self._logstd = self.add_weight(
            'logstd', shape, dtype=self._dtype, initializer=tf.constant_initializer(self._initial_logstd))
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        epsilon = tf.random_normal(tf.shape(inputs))
        return inputs + tf.cast(epsilon * tf.expand_dims(tf.exp(self._logstd), 0), inputs.dtype)

    @property
    def logstd(self) -> tf.Tensor:
        return self._logstd

    @property
    def std(self) -> tf.Tensor:
        return tf.exp(self._logstd)

    def get_config(self) -> Dict:
        config = {
            'noise_shape': self._noise_shape,
            'initial_logstd': self._initial_logstd
        }
        return config


class CouplingLayer(Model):

    def __init__(self, n_units, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.pred_log_s = Dense(n_units)
        self.pred_t = Dense(n_units)

    def call(self, inputs, reverse=False, **kwargs):
        inputs_a, inputs_b = inputs
        transform = self.layer(inputs_a, **kwargs)
        log_s = self.pred_log_s(transform)
        t = self.pred_t(transform)
        if reverse:
            b_transform = (inputs_b - t) / tf.exp(log_s)
            return b_transform

        b_transform = tf.exp(log_s) * inputs_b + t            
        return b_transform, log_s
