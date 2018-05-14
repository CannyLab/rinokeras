import copy
from functools import reduce
from operator import mul

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
            self._scope = tf.get_variable_scope() # do it like this because you could be inside another scope - this will give you the full scope path
            self.sy_obs = tf.placeholder(obs_dtype, (None,) + self._obs_shape, name='obs_placeholder')
            self._action, self._logits, self._value, self._layers, self._policy_scope = self._setup_agent(self.sy_obs, 'agent')
            self._model_vars = self._policy_scope.global_variables()

    def _setup_agent(self, img_in, scope):
        with tf.variable_scope(scope):
            layers = []
            embedding = self._embedding_network(img_in, layers)
            logits = self._action_logits(embedding, layers)
            value = self._value_function(embedding)
            if self._discrete:
                action = tf.argmax(logits, 1) if self._action_method == 'greedy' else tf.squeeze(tf.multinomial(logits, 1))
            else:
                action = logits if self._action_method == 'greedy' else logits
            return action, logits, value, layers, tf.get_variable_scope()

    def _embedding_network(self, img_in, layers, reuse=False):
        if img_in.dtype == tf.uint8:
            img_in = tf.cast(img_in, tf.float32) / 255.0

        if self._embedding_architecture is None:
            return img_in

        with tf.variable_scope('embedding', reuse=reuse):
            out = img_in
            func = slim.conv2d if self._use_conv else slim.fully_connected
            for layer in self._embedding_architecture:
                out = func(out, *layer)
                layers.append(tf.contrib.layers.flatten(out))
            return layers[-1]

    # TODO: add support for continuous action spaces
    def _action_logits(self, embedding, layers, reuse=False):
        if self._logit_architecture is None:
            raise ValueError("Received NoneType for action logit architecture.")

        with tf.variable_scope('logits', reuse=reuse):
            out = embedding
            for layer in self._logit_architecture:
                out = slim.fully_connected(out, *layer)
                layers.append(out)
            if self._discrete:
                logits = slim.fully_connected(out, self._ac_shape, activation_fn=None)
            else:
                ac_dim = reduce(mul, self._ac_shape)
                logits = slim.fully_connected(out, ac_dim, activation_fn=None)
                logits = tf.reshape(logits, self._ac_shape)
            layers.append(logits)
            return logits

    def _value_function(self, embedding, reuse=False):
        if self._value_architecture is None:
            return None

        with tf.variable_scope('value', reuse=reuse):
            out = embedding
            for layer in self._value_architecture:
                out = slim.fully_connected(out, *layer)
            value = slim.fully_connected(out, 1, activation_fn=None)
            value = tf.squeeze(value)
            return value

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




