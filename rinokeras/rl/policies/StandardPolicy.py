from operator import mul
from functools import reduce

import tensorflow as tf
import numpy as np

from typing import Tuple

from .Policy import Policy
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape
from rinokeras.common.layers import Stack, DenseStack
from rinokeras.common.distributions import CategoricalPd, DiagGaussianPd


class StandardPolicy(Model):

    def __init__(self,
                 action_shape: Tuple[int, ...],
                 action_space: str,
                 embedding_model: Model,
                 model_dim: int = 64,
                 n_layers_logits: int = 1,
                 n_layers_value: int = 1,
                 take_greedy_actions: bool = False,
                 initial_logstd: float = 0,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        if not isinstance(action_shape, (tuple, list)):
            raise TypeError("Expected tuple or list for action shape, received {}".format(type(action_shape)))
        if action_space not in ['discrete', 'continuous']:
            raise ValueError("action_space must be one of <discrete, continuous>, received {}".format(action_space))

        self.action_shape = action_shape
        self.action_space = action_space
        self.model_dim = model_dim
        self.n_layers_logits = n_layers_logits
        self.n_layers_value = n_layers_value
        self.take_greedy_actions = take_greedy_actions
        self.initial_logstd = initial_logstd
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        self.embedding_model = embedding_model
        self.logits_function = self._setup_logits_function()
        self.value_function = self._setup_value_function()
        self.action_distribution = CategoricalPd(name='action') if action_space == 'discrete' \
            else DiagGaussianPd(action_shape, initial_logstd=initial_logstd, name='action')

    def _setup_logits_function(self, activation=None):
        ac_dim = reduce(mul, self.action_shape)

        logits_function = Stack(name='logits')
        logits_function.add(
            DenseStack(self.n_layers_logits * [self.model_dim] + [ac_dim], output_activation=activation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer))
        logits_function.add(Reshape(self.action_shape))
        return logits_function

    def _setup_value_function(self):
        value_function = DenseStack(
            self.n_layers_value * [self.model_dim] + [1], output_activation=None,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer)
        return value_function

    def call(self, obs, training=False):
        self._obs = obs

        if self._obs.shape[1].value is None:
            bs, seqlen = tf.shape(obs)[0], tf.shape(obs)[1]
            obs = tf.reshape(obs, (bs * seqlen, obs.shape[-1]))

        embedding = self.embedding_model(obs)
        logits = self.logits_function(embedding)

        value = tf.squeeze(self.value_function(embedding), -1)
        action = self.action_distribution(logits, greedy=self.take_greedy_actions)

        if self._obs.shape[1].value is None:
            value = tf.reshape(value, (bs, seqlen))
            logits = tf.reshape(logits, (bs, seqlen) + self.action_shape)

        self._action = action
        self._value = value

        if training:
            return logits, value
        else:
            return action

    def predict(self, obs):

        if tf.executing_eagerly():
            obs = tf.cast(tf.constant(obs), tf.float32)
            action = self(obs, training=False).numpy()
        else:
            if not self.built:
                raise RuntimeError("Policy is not built, please call the policy before running predict.")
            if self._obs.shape[1].value is None:
                obs = obs[:, None]  # Expand the time dimension
            sess = self._get_session()
            action = sess.run(self._action, feed_dict={self._obs: obs})[0]

        return action

    def logp_actions(self, logits, actions):
        return self.action_distribution.logp_actions(logits, actions)

    def entropy(self, logits):
        return self.action_distribution.entropy(logits)

    def _get_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("This method must be run inside a tf.Session context")
        return sess

    def get_config(self):
        config = {
            'action_shape': self.action_shape,
            'action_space': self.action_space,
            'embedding_model': self.embedding_model.__class__.from_config(self.embedding_model.get_config()),
            'model_dim': self.model_dim,
            'n_layers_logits': self.n_layers_logits,
            'n_layers_value': self.n_layers_value,
            'take_greedy_actions': self.take_greedy_actions,
            'initial_logstd': self.initial_logstd
        }
        return config

    # TODO: This doesn't actually match how keras does from config I think
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    def clone(self):
        newcls = self.__class__.from_config(self.get_config())
        newcls.build(self.input_shape)
        newcls.set_weights(self.get_weights())
        return newcls

    def clear_memory(self) -> None:
        pass
