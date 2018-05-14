from functools import reduce
from operator import mul

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .StandardPolicy import StandardPolicy

class LSTMPolicy(StandardPolicy):

    def _setup_placeholders(self):
        self.sy_obs = tf.plaeholder(self._obs_dtype, (None, None) + self._obs_shape, name='obs_placeholder')

    def _setup_embedding_network(self, reuse=False):
        batch_size = tf.shape(self.sy_obs)[0]
        num_timesteps = tf.shape(self.sy_obs)[1]

        if self.sy_obs.dtype == tf.uint8:
            self.sy_obs = tf.cast(self.sy_obs, tf.float32) / 255.0

        if self._embedding_architecture is not None:
            with tf.variable_scope('embedding', reuse=reuse):
                out = self.sy_obs
                func = slim.conv2d if self._use_conv else slim.fully_connected
                for layer in self._embedding_architecture:
                    out = func(out, *layer)
                    self._layers.append(tf.contrib.layers.flatten(out))
                out = self._layers[-1]
        else:
            out = self.sy_obs

        embedding_size = tf.shape(out)[1]
        out = tf.reshape(out, (batch_size, num_timesteps, embedding_size))

        with tf.variable_scope('lstm', reuse=reuse):
            lstm = rnn.rnn_cell.BasicLSTMCell(self._lstm_cell_size)
            self._state_size = lstm.state_size

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self._state_init = (c_init, h_init)
            self._state_curr = self._state_init
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
            self._state_in = (c_in, h_in)

            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, out, initial_state=state_in, dtype=tf.float32)

            lstm_c, lstm_h = lstm_state
            self._embedding = tf.reshape(lstm_outputs, [-1, self._lstm_cell_size])

            # All this stuff is only needed when collecting rollouts, so can hardcode [0, :]
            self._state_out = [lstm_c[:1,:], lstm_h[:1,:]] # doing it like this does keepdims automatically I think

    def predict(self, obs, return_activations=False):
        sess = self._get_session()
        to_return = [self._action, self._state_out[0], self._state_out[1]]
        if return_activations:
            to_return.append(self._layers)
        feed_dict = {self.sy_obs : obs,
                        self._state_in[0] : self._state_curr[0],
                        self._state_in[1] : self._state_curr[1]}
        to_return = sess.run(to_return, feed_dict=feed_dict)
        self._state_curr = to_return[1:3]
        return to_return[0] if not return_activations else to_return

    def clear_memory(self):
        self._state_curr = self._state_init



        
