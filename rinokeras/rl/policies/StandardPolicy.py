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
from rinokeras.common.layers import Stack, DenseStack
from rinokeras.common.distributions import CategoricalPd, DiagGaussianPd

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
                 extra_tensors: Optional[Dict[str, tf.Tensor]] = None,
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
        self.state = tf.constant([])
        self.normalize_observations = normalize_observations

        if normalize_observations:
            self.batch_norm = BatchNormalization(center=False, scale=False)

        if extra_tensors is not None:
            self.__dict__.update(extra_tensors)

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
        self.X = obs

        if self.normalize_observations and obs.dtype == tf.float32:
            obs = self.batch_norm(obs)
            obs = tf.clip_by_value(obs, -5.0, 5.0)

        obs = self.encode_observation(self.obs_space, obs)

        if self._obs.shape[1].value is None:
            bs, seqlen = tf.shape(obs)[0], tf.shape(obs)[1]
            remaining_shape = obs.shape[2:].as_list()
            obs = tf.reshape(obs, [bs * seqlen] + remaining_shape)

        embedding = self.embedding_model(obs)
        self.embedding = embedding
        logits = self.logits_function(embedding)

        value = tf.squeeze(self.value_function(embedding), -1)
        action = self.pd(logits, greedy=self.take_greedy_actions)

        if self._obs.shape[1].value is None:
            value = tf.reshape(value, (bs, seqlen))
            logits = tf.reshape(logits, (bs, seqlen) + self.act_shape)

        self._logits = logits
        self.action = action
        self.vf = value
        self.neglogpac = self.neglogp(action)

        return logits, value, action

    def predict(self, obs):

        if tf.executing_eagerly():
            raise RuntimeError(
                "Modifications for compatibility with openai baselines code destroyed eager execution ability."
                " So unfortunately you can't run this in eager.")
            # obs = tf.cast(tf.constant(obs), tf.float32)
            # action = self(obs, training=False).numpy()
        else:
            if not self.built:
                raise RuntimeError("Policy is not built, please call the policy before running predict.")
            if self._obs.shape[1].value is None:
                obs = obs[:, None]  # Expand the time dimension
            sess = self._get_session()
            action = sess.run(self._action, feed_dict={self._obs: obs})[0]

        return action

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask
            (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
            (action, value estimate, next state,
                negative log likelihood of the action under current policy parameters) tuple
        """
        action, value, state, neglogp = self._run_tensors(
            [self.action, self.vf, self.state, self.neglogpac], observation, **extra_feed)

        if state.size == 0:
            state = None

        return action, value, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask
            (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._run_tensors(self.vf, ob, *args, **kwargs)

    def logp_actions(self, actions):
        return self.pd.logp_actions(actions)

    def neglogp(self, actions):
        return - self.logp_actions(actions)

    def entropy(self):
        return self.pd.entropy()

    def _run_tensors(self, variables, observation, **extra_feed):
        sess = K.get_session()
        feed_dict = {self._obs: adjust_shape(self._obs, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

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

    def encode_observation(self, ob_space, placeholder):
        '''
        Encode input in the way that is appropriate to the observation space

        Parameters:
        ----------

        ob_space: gym.Space             observation space

        placeholder: tf.placeholder     observation input placeholder
        '''
        if isinstance(ob_space, gym.spaces.Discrete):
            return tf.to_float(tf.one_hot(placeholder, ob_space.n))
        elif isinstance(ob_space, gym.spaces.Box):
            return tf.to_float(placeholder)
        elif isinstance(ob_space, gym.spaces.MultiDiscrete):
            placeholder = tf.cast(placeholder, tf.int32)
            one_hots = [tf.to_float(tf.one_hot(placeholder[..., i], ob_space.nvec[i]))
                        for i in range(placeholder.shape[-1])]
            return tf.concat(one_hots, axis=-1)
        else:
            raise NotImplementedError
