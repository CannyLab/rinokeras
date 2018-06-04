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
                        value_architecture=[(64,), (64,)],
                        initial_logstd=0):
        super().__init__(obs_shape, ac_shape, discrete, scope)
        self._use_conv = use_conv
        self._embedding_architecture = embedding_architecture
        self._logit_architecture = logit_architecture
        self._value_architecture = value_architecture
        self._action_method = action_method
        self._obs_dtype = obs_dtype
        self._initial_logstd = initial_logstd
        if action_method not in ['greedy', 'sample']:
            raise ValueError("Unrecognized action method")

        with tf.variable_scope(scope):
            self._setup_placeholders()
            self._layers = {}

            self._setup_agent()
            self._setup_action()

            self._scope = tf.get_variable_scope() # do it like this because you could be inside another scope - this will give you the full scope path
            self._model_vars = self._scope.global_variables()

    def _setup_placeholders(self):
        self.sy_obs = tf.placeholder(self._obs_dtype, (None,) + self._obs_shape, name='obs_placeholder')

    def _setup_agent(self):
        self._embedding = self._setup_embedding_network(self.sy_obs, self._layers)
        self._logits = self._setup_action_logits(self._embedding, self._layers)
        if self._value_architecture is not None:
            value_embedding = self._setup_embedding_network(self.sy_obs, {}, scope='value_embedding')
            self._value = self._setup_value_function(value_embedding, {})

    def _setup_action(self):
        if self._discrete:
            action = tf.argmax(self._logits, 1) if self._action_method == 'greedy' else tf.multinomial(self._logits, 1)
            self._action = tf.squeeze(action)
            # self._neglogpac = tf.softmax(self._logits)
        else:
            if self._action_method == 'greedy':
                self._action = self._logits
            else:
                self._log_std = tf.get_variable('log_std', shape=self._ac_shape, dtype=tf.float32, initializer=tf.constant_initializer(self._initial_logstd))
                self._std = tf.exp(self._log_std)
                epsilon = tf.random_normal(tf.shape(self._logits))
                self._action = self._logits + epsilon * self._std
                # norm_diff = 0.5 * tf.reduce_sum(tf.square(self._action - self._logits) / self._std, np.arange(1, len(self._ac_shape) + 1))
                # neg_logprobs = norm_diff + 0.5 * np.log(2.0 * np.pi) * reduce(mul, self._ac_shape) + tf.reduce_sum(self._log_std)
                # self._neglogpac = neg_logprobs

                # self._neglogpac = 0.5 * tf.reduce_sum(tf.square((self._action - self._logits) / self._std), axis=-1) \
                #    + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self._action)[-1]) \
                #    + tf.reduce_sum(self._log_std, axis=-1)

    def _setup_embedding_network(self, inputs, layers, reuse=False, scope='embedding'):
        layers[scope] = []
        embedding = inputs
        if self.sy_obs.dtype == tf.uint8:
            embedding = tf.cast(embedding, tf.float32) / 255.0

        if self._embedding_architecture is None:
            layers[scope].append(tf.contrib.layers.flatten(embedding))
        else:
            with tf.variable_scope(scope, reuse=reuse):
                func = slim.conv2d if self._use_conv else slim.fully_connected
                for layer in self._embedding_architecture:
                    embedding = func(embedding, *layer)
                    layers[scope].append(tf.contrib.layers.flatten(embedding))
        return layers[scope][-1]

    def _setup_action_logits(self, inputs, layers, reuse=False, scope='logits'):
        if self._logit_architecture is None:
            raise TypeError("Received NoneType for action logit architecture.")
        layers[scope] = []

        with tf.variable_scope(scope, reuse=reuse):
            out = inputs
            for layer in self._logit_architecture:
                out = slim.fully_connected(out, *layer)
                layers[scope].append(out)
            if self._discrete:
                logits = slim.fully_connected(out, self._ac_shape, activation_fn=None)
            else:
                ac_dim = self._ac_shape 
                if isinstance(ac_dim, collections.Iterable):
                    ac_dim = reduce(mul, self._ac_shape)
                logits = slim.fully_connected(out, ac_dim, activation_fn=None)
                logits = tf.reshape(logits, (-1,) + self._ac_shape)
            layers[scope].append(logits)
            return logits

    def _setup_value_function(self, inputs, layers, reuse=False):
        layers['value'] = []
        if self._value_architecture is None:
            layers['value'].append(None)
        else:
            with tf.variable_scope('value', reuse=reuse):
                out = inputs
                for layer in self._value_architecture:
                    out = slim.fully_connected(out, *layer)
                    layers['value'].append(out)
                value = slim.fully_connected(out, 1, activation_fn=None)
                # value = tf.squeeze(value)
                value = value[:,0]
                layers['value'].append(value)
        return layers['value'][-1]

    def get_neg_logp_actions(self, actions):
        if self._discrete:
            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=self._logits)
        else:
            return 0.5 * tf.reduce_sum(tf.square((actions - self._logits) / self._std), np.arange(1, len(self._ac_shape) + 1)) \
                    + 0.5 * np.log(2.0 * np.pi) * reduce(mul, self._ac_shape) + tf.reduce_sum(self._log_std)

    def entropy(self):
        if self._discrete:
            probs = tf.nn.softmax(self._logits)
            logprobs = tf.log(probs)
            return -tf.reduce_sum(tf.multiply(probs, logprobs), 1)
        else:
            return tf.reduce_sum(self._log_std + 0.5 * np.log(2.0 * np.pi * np.e))

    def predict(self, obs, return_activations=False):
        sess = self._get_session()
        to_return = [self._action]
        if return_activations:
            to_return.append(self._layers)
        to_return = sess.run(to_return, feed_dict={self.sy_obs : obs})
        return to_return[0] if not return_activations else to_return

    def step(self, obs, states, dones):
        sess = self._get_session()
        actions, values, neglogpac = sess.run([self._action, self._value, self._neglogpac], feed_dict = {self.sy_obs : obs})
        return actions, values, None, neglogpac

    def predict_value(self, obs, dummy, dummy1):
        sess = self._get_session()
        return sess.run(self._value, feed_dict={self.sy_obs : obs})

    def make_copy(self, scope):
        return self.__class__(self._obs_shape,
                                self._ac_shape,
                                self._discrete,
                                scope,
                                self._obs_dtype,
                                self._action_method,
                                self._use_conv,
                                self._embedding_architecture,
                                self._logit_architecture,
                                self._value_architecture,
                                self._initial_logstd)

    def clear_memory(self):
        return




