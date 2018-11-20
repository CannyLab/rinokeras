from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Optional, Tuple

import tensorflow as tf
import numpy as np
from rinokeras.common.layers import RandomGaussNoise


class Pd(tf.keras.Model, ABC):

    @abstractmethod
    def call(self, logits, greedy=False):
        return NotImplemented

    @abstractmethod
    def logp_action(self, logits, action):
        return NotImplemented

    def prob_action(self, logits, action):
        return tf.exp(self.logp(logits, action))

    @abstractmethod
    def entropy(self, logits):
        return NotImplemented


class CategoricalPd(Pd):

    def call(self, logits, greedy=False):
        return tf.squeeze(tf.argmax(logits, -1)) if greedy else tf.squeeze(tf.multinomial(logits, -1))

    def logp_action(self, logits, action):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action, logits=logits)

    def entropy(self, logits):
        # Have to calculate these manually b/c logp_action provides probabilities
        # for a specific action.
        probs = tf.nn.softmax(logits)
        logprobs = tf.log(probs)
        return - tf.reduce_mean(probs * logprobs, 1)


class DiagGaussianPd(tf.keras.layers.Layer):

    def __init__(self, noise_shape: Optional[Tuple[int]] = None, initial_logstd: float = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._noise_shape = noise_shape
        self._initial_logstd = initial_logstd

    def build(self, input_shape):
        self._add_noise = RandomGaussNoise(self._noise_shape, self._initial_logstd)
        super().build(input_shape)

    def call(self, logits, greedy=False):
        return logits if greedy else self._add_noise(logits)

    def logp_action(self, logits, action):
        std = self._add_noise.std
        logstd = self._add_noise.logstd
        sqdiff = tf.squared_difference(action, logits)
        reduction_axes = np.arange(1, len(logits.shape))
        divconst = np.log(2.0 * np.pi) * tf.cast(tf.prod(tf.shape(action)[1:]), tf.float32) + tf.reduce_sum(logstd)
        return -0.5 * (tf.reduce_sum(sqdiff / std, reduction_axes) + divconst)

    def entropy(self, logits):
        return tf.reduce_sum(self._add_noise.logstd + 0.5 * np.log(2.0 * np.pi * np.e))


__all__ = ['Pd', 'CategoricalPd', 'DiagGaussianPd']
