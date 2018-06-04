import collections
from operator import mul
from functools import reduce

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

from .layers import Stack, Conv2DStack, DenseStack, RandomNoise

class StandardPolicy(tf.keras.Model):

    def __init__(self, 
                 obs_shape,
                 ac_shape, 
                 discrete,
                 action_method='greedy',
                 use_conv=False,
                 embedding_architecture=[64, 64],
                 logit_architecture=[64, 64],
                 value_architecture=[64],
                 initial_logstd=0):

        super().__init__()
        self._obs_shape = obs_shape
        self._ac_shape = ac_shape
        if not isinstance(self._ac_shape, collections.Iterable):
            self._ac_shape = (ac_shape,)
        self._discrete = discrete
        self._action_method = action_method
        self._use_conv = use_conv
        self._embedding_architecture = embedding_architecture
        self._logit_architecture = logit_architecture
        self._value_architecture = value_architecture
        self._initial_logstd = initial_logstd

        self._built = False
        self._setup_agent()
        self.build()

    def _setup_agent(self):
        self._embedding_function = self._setup_embedding_function()
        self._logits_function = self._setup_logit_function()
        self._value_function = self._setup_value_function()
        self._action_function = self._setup_action_function()

    def _setup_action_function(self):
        if self._discrete and self._action_method == 'greedy':
            def get_action(logits):
                return tf.squeeze(tf.argmax(logits, 1))

            return get_action
        
        elif self._discrete and self._action_method == 'sample':
            def get_action(logits):
                return tf.squeeze(tf.multinomial(logits, 1))

            return get_action

        elif not self._discrete and self._action_method == 'greedy':
            def get_action(logits):
                return tf.squeeze(logits)

            return get_action

        elif not self._discrete and self._action_method == 'sample':
            get_action = RandomNoise(self._ac_shape, self._initial_logstd)
            # self._logstd = tfe.Variable(np.zeros(self._ac_shape) + self._initial_logstd, dtype=tf.float32)
            # self.trainable_variables.append(self._logstd)
            
            # def get_action(logits):
                # epsilon = tf.random_normal(self._ac_shape)
                # return logits + epsilon * self.std

            return get_action

    def _setup_embedding_function(self):
        return Conv2DStack(self._embedding_architecture) if self._use_conv \
                    else DenseStack(self._embedding_architecture)


    def _setup_logit_function(self, activation=None):
        ac_dim = reduce(mul, self._ac_shape)

        logit_function = Stack()
        if self._logit_architecture is not None:
            logit_function.add(DenseStack(self._logit_architecture))
        logit_function.add(tf.keras.layers.Dense(ac_dim))
        if activation is not None:
            logit_function.add(tf.keras.layers.Activation(activation))
        logit_function.add(tf.keras.layers.Reshape(self._ac_shape))
        return logit_function

    def _setup_value_function(self):
        if self._value_architecture is None:
            return None
        value_function = Stack()
        value_function.add(DenseStack(self._value_architecture))
        value_function.add(tf.keras.layers.Dense(1))
        return value_function 

    def build(self):
        if not self._built:
            dummy_obs = tf.zeros((1,) + self._obs_shape, dtype=tf.float32)
            
            embedding = self._embedding_function(dummy_obs)
            logits = self._logits_function(embedding)
            value = self._value_function(embedding)
            action = self._action_function(logits)
            self._built = True

    def call(self, obs, is_training=False):
        if not self._built:
            self.build()
        if obs.dtype == np.uint8:
            obs = np.asarray(obs, np.float32) / 255
        embedding = self._embedding_function(obs)
        logits = self._logits_function(embedding)
        if is_training:
            value = self._value_function(embedding)
            return logits, tf.squeeze(value)
        else:
            action = self._action_function(logits)
            return action.numpy()

    def predict(self, obs, return_activations=False):
        return self.call(obs, is_training=False)

    def get_neg_logp_actions(self, logits, actions):
        if self._discrete:
            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
        else:
            return 0.5 * tf.reduce_sum(tf.square((actions - logits) / self._action_function.std), np.arange(1, len(self._ac_shape) + 1)) \
                    + 0.5 * np.log(2.0 * np.pi) * reduce(mul, self._ac_shape) + tf.reduce_sum(self._action_function.logstd)

    def entropy(self, logits):
        if self._discrete:
            probs = tf.nn.softmax(logits)
            logprobs = tf.log(probs)
            return - tf.reduce_sum(probs * logprobs, 1)
        else:
            return tf.reduce_sum(self._action_function.logstd + 0.5 * np.log(2.0 * np.pi * np.e))

    def clear_memory(self):
        return

    def make_copy(self):
        return self.__class__(self._obs_shape,
                                self._ac_shape,
                                self._discrete,
                                self._action_method,
                                self._use_conv,
                                self._embedding_architecture,
                                self._logit_architecture,
                                self._value_architecture,
                                self._initial_logstd)
