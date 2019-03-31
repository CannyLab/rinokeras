from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Optional, Tuple

import tensorflow as tf
import numpy as np
from rinokeras.core.v1x.common.layers import RandomGaussNoise


class Pd(tf.keras.Model, ABC):

    @abstractmethod
    def call(self, logits, greedy=False):
        return NotImplemented

    @abstractmethod
    def logp_actions(self, actions):
        return NotImplemented

    def neglogp(self, actions):
        return -self.logp_actions(actions)

    def prob_actions(self, actions):
        return tf.exp(self.logp(self._logits, actions))

    @abstractmethod
    def entropy(self):
        return NotImplemented


class CategoricalPd(Pd):

    def call(self, logits, greedy=False):
        self._logits = logits
        if greedy:
            action = tf.argmax(logits, -1)
        else:
            u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
            action = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
            # if logits.shape.ndims == 2:
                # action = tf.squeeze(tf.multinomial(logits, 1), -1)
            # else:

                # fixed_shapes = logits.shape.as_list()[:-1]
                # variable_shapes = tf.shape(logits)[:-1]
                # action_shape = [fs if fs is not None else variable_shapes[i] for i, fs in enumerate(fixed_shapes)]

                # logits = tf.reshape(logits, (-1, logits.shape[-1]))
                # action = tf.squeeze(tf.multinomial(logits, 1), -1)
                # action = tf.reshape(action, action_shape)

        return action

    def logp_actions(self, actions):
        probs = tf.nn.softmax(self._logits - tf.reduce_max(self._logits, -1, keepdims=True))
        indices = tf.one_hot(actions, depth=probs.shape[-1])
        prob_act = tf.reduce_max(probs * indices, -1)
        logp_act = tf.log(prob_act + 1e-8)
        return logp_act
        # return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            # labels=actions, logits=logits - tf.reduce_max(logits, -1, keepdims=True))

    def entropy(self):
        # Have to calculate these manually b/c logp_action provides probabilities
        # for a specific action
        a0 = self._logits - tf.reduce_max(self._logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        x0 = p0 * (tf.log(z0) - a0)
        x1 = tf.reduce_sum(x0, axis=-1)
        return x1


class DiagGaussianPd(Pd):

    def __init__(self,
                 action_shape: Tuple[int, ...],
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
        self._logits = logits
        return logits if greedy else self._add_noise(logits)

    def logp_actions(self, actions):
        sqdiff = tf.squared_difference(actions, self._logits)
        reduction_axes = np.arange(-1, -self._ndim_action - 1, -1)
        divconst = np.log(2.0 * np.pi) * tf.cast(tf.reduce_prod(tf.shape(actions)[1:]), tf.float32) + tf.reduce_sum(self.logstd)
        return -0.5 * (tf.reduce_sum(sqdiff / self.std, reduction_axes) + divconst)

    def entropy(self):
        x0 = self._add_noise.logstd + 0.5 * np.log(2.0 * np.pi * np.e)
        return tf.reduce_sum(x0)

    @property
    def std(self):
        return self._add_noise.std

    @property
    def logstd(self):
        return self._add_noise.logstd


__all__ = ['Pd', 'CategoricalPd', 'DiagGaussianPd']
