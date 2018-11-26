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
    def logp_actions(self, logits, actions):
        return NotImplemented

    def prob_actions(self, logits, actions):
        return tf.exp(self.logp(logits, actions))

    @abstractmethod
    def entropy(self, logits):
        return NotImplemented


class CategoricalPd(Pd):

    def call(self, logits, greedy=False):
        if greedy:
            action = tf.argmax(logits, -1)
        else:
            if logits.shape.ndims == 2:
                action = tf.squeeze(tf.multinomial(logits, 1), -1)
            else:

                fixed_shapes = logits.shape.as_list()[:-1]
                variable_shapes = tf.shape(logits)[:-1]
                action_shape = [fs if fs is not None else variable_shapes[i] for i, fs in enumerate(fixed_shapes)]

                logits = tf.reshape(logits, (-1, logits.shape[-1]))
                action = tf.squeeze(tf.multinomial(logits, 1), -1)
                action = tf.reshape(action, action_shape)

        return action

    def logp_actions(self, logits, actions):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)

    def entropy(self, logits):
        # Have to calculate these manually b/c logp_action provides probabilities
        # for a specific action.
        probs = tf.nn.softmax(logits)
        logprobs = tf.log(probs)
        return - tf.reduce_mean(probs * logprobs, axis=-1)


class DiagGaussianPd(Pd):

    def __init__(self,
                 action_shape: Tuple[int],
                 noise_shape: Optional[Tuple[int]] = None,
                 initial_logstd: float = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._action_shape = action_shape
        self._ndim_action = len(action_shape)
        self._noise_shape = noise_shape
        self._initial_logstd = initial_logstd

    def build(self, input_shape):
        self._add_noise = RandomGaussNoise(self._noise_shape, self._initial_logstd)
        # super().build(input_shape)

    def call(self, logits, greedy=False):
        return logits if greedy else self._add_noise(logits)

    def logp_actions(self, logits, actions):
        sqdiff = tf.squared_difference(actions, logits)
        reduction_axes = np.arange(-1, -self._ndim_action - 1, -1)
        divconst = np.log(2.0 * np.pi) * tf.cast(tf.reduce_prod(tf.shape(actions)[1:]), tf.float32) + tf.reduce_sum(self.logstd)
        return -0.5 * (tf.reduce_sum(sqdiff / self.std, reduction_axes) + divconst)

    def entropy(self, logits):
        return tf.reduce_sum(self._add_noise.logstd + 0.5 * np.log(2.0 * np.pi * np.e))

    @property
    def std(self):
        return self._add_noise.std

    @property
    def logstd(self):
        return self._add_noise.logstd


__all__ = ['Pd', 'CategoricalPd', 'DiagGaussianPd']
