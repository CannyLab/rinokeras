from functools import reduce
from operator import mul
import collections

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .TFPolicy import TFPolicy

class StandardPolicy(TFPolicy):

    def __init__(self, obs_shape, 
                        ac_shape, 
                        discrete, 
                        scope='agent', 
                        obs_dtype=tf.float32,
                        action_method='greedy',
                        use_conv=False,
                        embedding_architecture=[(64,), (64,)],
                        logit_architecture=[(64,), (64,)],
                        value_architecture=[(64,), (64,)]):
        super().__init__(obs_shape, ac_shape, discrete, scope)
        self._use_conv = use_conv
        self._embedding_architecture = embedding_architecture
        self._logit_architecture = logit_architecture
        self._value_architecture = value_architecture
        self._action_method = action_method
        self._obs_dtype = obs_dtype
        if action_method not in ['greedy', 'sample']:
            raise ValueError("Unrecognized action method")

        with tf.variable_scope(scope):
            self._setup_placeholders()
            self._layers = []

            self._setup_embedding_network()
            self._setup_action_logits()
            self._setup_value_function()

            if self._discrete:
                self._action = tf.argmax(self._logits, 1) if self._action_method == 'greedy' else tf.squeeze(tf.multinomial(self._logits, 1))
            else:
                self._log_std = tf.get_variable('log_std', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(-1))
                self._action = self._logits if self._action_method == 'greedy' else self._logits + tf.random_normal(tf.shape(self._logits), 0, tf.exp(self._log_std))
                self._action = self._action[0] # only needed during inference

            self._scope = tf.get_variable_scope() # do it like this because you could be inside another scope - this will give you the full scope path
            self._model_vars = self._scope.global_variables()

    def _setup_placeholders(self):
        self.sy_obs = tf.placeholder(self._obs_dtype, (None,) + self._obs_shape, name='obs_placeholder')

    def _setup_embedding_network(self, reuse=False):
        if self.sy_obs.dtype == tf.uint8:
            self.sy_obs = tf.cast(self.sy_obs, tf.float32) / 255.0

        if self._embedding_architecture is None:
            self._embedding = self.sy_obs
            return

        with tf.variable_scope('embedding', reuse=reuse):
            out = self.sy_obs
            func = slim.conv2d if self._use_conv else slim.fully_connected
            for layer in self._embedding_architecture:
                out = func(out, *layer)
                self._layers.append(tf.contrib.layers.flatten(out))
            self._embedding = self._layers[-1]

    def _setup_action_logits(self, reuse=False):
        if self._logit_architecture is None:
            raise ValueError("Received NoneType for action logit architecture.")

        with tf.variable_scope('logits', reuse=reuse):
            out = self._embedding
            for layer in self._logit_architecture:
                out = slim.fully_connected(out, *layer)
                self._layers.append(out)
            if self._discrete:
                logits = slim.fully_connected(out, self._ac_shape, activation_fn=None)
            else:
                ac_dim = self._ac_shape 
                if isinstance(ac_dim, collections.Iterable):
                    ac_dim = reduce(mul, self._ac_shape)
                logits = slim.fully_connected(out, ac_dim, activation_fn=None)
                logits = tf.reshape(logits, (-1,) + self._ac_shape)
            self._layers.append(logits)
            self._logits = logits

    def _setup_value_function(self, reuse=False):
        if self._value_architecture is None:
            self._value = None
            return

        with tf.variable_scope('value', reuse=reuse):
            out = self._embedding
            for layer in self._value_architecture:
                out = slim.fully_connected(out, *layer)
            value = slim.fully_connected(out, 1, activation_fn=None)
            value = tf.squeeze(value)
            self._value = value

    def predict(self, obs, return_activations=False):
        sess = self._get_session()
        to_return = [self._action]
        if return_activations:
            to_return.append(self._layers)
        to_return = sess.run(to_return, feed_dict={self.sy_obs : obs})
        return to_return[0] if not return_activations else to_return

    def make_copy(self, scope):
        return StandardPolicy(self._obs_shape,
                                self._ac_shape,
                                self._discrete,
                                scope,
                                self._obs_dtype,
                                self._action_method,
                                self._use_conv,
                                self._embedding_architecture,
                                self._logit_architecture,
                                self._value_architecture)




