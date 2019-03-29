from operator import mul
from functools import reduce
from collections import namedtuple
from typing import Optional, Dict

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from typing import Tuple

from .Policy import Policy
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, BatchNormalization
from rinokeras.core.v1x.common.layers import Stack, DenseStack
from rinokeras.core.v1x.common.distributions import CategoricalPd, DiagGaussianPd
from rinokeras.core.v1x.utils import get_shape

from baselines.common.tf_util import adjust_shape

import gym


class StandardPolicy(Model):

    def __init__(self,
                 obs_space: gym.Space,
                 act_space: gym.Space,
                 embedding_model: Model,
                 model_dim: int = 64,
                 n_layers_logits: int = 1,
                 n_layers_value: int = 1,
                 take_greedy_actions: bool = False,
                 initial_logstd: float = 0,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 normalize_observations: bool = False,
                 **kwargs) -> None:

        super().__init__(**kwargs)

        self.obs_space = obs_space
        self.act_space = act_space
        self.act_shape = (act_space.n,) if isinstance(act_space, gym.spaces.Discrete) else act_space.shape
        self.model_dim = model_dim
        self.n_layers_logits = n_layers_logits
        self.n_layers_value = n_layers_value
        self.take_greedy_actions = take_greedy_actions
        self.initial_logstd = initial_logstd
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.initial_state = None
        self.normalize_observations = normalize_observations

        if normalize_observations:
            self.batch_norm = BatchNormalization(center=False, scale=False)

        self.embedding_model = embedding_model
        self.logits_function = self._setup_logits_function()
        self.value_function = self._setup_value_function()
        self.pd = CategoricalPd(name='action') if isinstance(act_space, gym.spaces.Discrete) \
            else DiagGaussianPd(act_space.shape, initial_logstd=initial_logstd, name='action')

    def _setup_logits_function(self, activation=None):
        ac_dim = reduce(mul, self.act_shape)

        logits_function = Stack(name='logits')
        logits_function.add(
            DenseStack(self.n_layers_logits * [self.model_dim] + [ac_dim], output_activation=activation,
                       kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                       activity_regularizer=self.activity_regularizer,
                       kernel_initializer=tf.keras.initializers.Orthogonal()))
        logits_function.add(Reshape(self.act_shape))
        return logits_function

    def _setup_value_function(self):
        value_function = DenseStack(
            self.n_layers_value * [self.model_dim] + [1], output_activation=None,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer)
        return value_function

    def call(self, obs, training=False):
        self._obs = obs

        if self.normalize_observations and obs.dtype == tf.float32:
            obs = self.batch_norm(obs)
            obs = tf.clip_by_value(obs, -5.0, 5.0)

        embedding = self.embedding_model(obs)
        state = tf.constant([])

        logits = self.logits_function(embedding)

        value = tf.squeeze(self.value_function(embedding), -1)

        action = self.pd(logits, greedy=self.take_greedy_actions)
        neglogpac = self.neglogp(action)

        return {'latent': embedding,
                'q': logits,
                'vf': value,
                'state': state,
                'action': action,
                'neglogp': neglogpac}

    def logp_actions(self, actions):
        return self.pd.logp_actions(actions)

    def neglogp(self, actions):
        return - self.logp_actions(actions)

    def entropy(self):
        return self.pd.entropy()

    def get_config(self):
        config = {
            'action_shape': self.act_shape,
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
