from typing import Tuple
from functools import reduce
from operator import mul

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Input, Reshape
import tensorflow.keras.backend as K
import numpy as np

from rinokeras.common.layers import Stack, DenseStack
from .StandardPolicy import StandardPolicy


class LSTMPolicy(StandardPolicy):

    def __init__(self,
                 action_shape: Tuple[int, ...],
                 action_space: str,
                 embedding_model: Model,
                 model_dim: int = 64,
                 n_layers_logits: int = 1,
                 n_layers_value: int = 1,
                 lstm_cell_size: int = 256,
                 take_greedy_actions: bool = False,
                 initial_logstd: float = 0,
                 **kwargs) -> None:

        super().__init__(
            action_shape, action_space, embedding_model, model_dim, n_layers_logits,
            n_layers_value, take_greedy_actions, initial_logstd, **kwargs)
        self.lstm_cell_size = lstm_cell_size

        self.memory_function = self._setup_memory_function()
        if not tf.executing_eagerly():
            self._memory_in = (Input((lstm_cell_size,)), Input((lstm_cell_size,)))
        self._current_memory = None

    def _setup_memory_function(self):
        return LSTM(self.lstm_cell_size, return_sequences=True, return_state=True)

    def call(self, obs, training=False):

        self._obs = obs

        bs, seqlen = tf.shape(obs)[0], tf.shape(obs)[1]
        obs = tf.reshape(obs, (bs * seqlen, obs.shape[-1]))

        embedding = self.embedding_model(obs)

        embedding = tf.reshape(embedding, (bs, seqlen, embedding.shape[-1]))
        memory_out, memory_h, memory_c = self.memory_function(
            embedding, initial_state=K.in_train_phase(None, self._memory_in, training=training))
        memory_state = (memory_h, memory_c)
        memory_out = tf.reshape(memory_out, (bs * seqlen, memory_out.shape[-1]))

        logits = self.logits_function(memory_out)

        value = tf.squeeze(self.value_function(memory_out), -1)
        action = self.action_distribution(logits, greedy=self.take_greedy_actions)

        value = tf.reshape(value, (bs, seqlen))
        logits = tf.reshape(logits, (bs, seqlen) + self.action_shape)

        self._action = action
        self._value = value
        self._memory_state = memory_state

        if training:
            return logits, value
        else:
            return action, memory_state

    def predict(self, obs):
        if not self.built:
            raise RuntimeError("Policy is not built, please call the policy before running predict.")

        # expand time dimension of observation
        obs = obs[:, None]

        if tf.executing_eagerly():
            self._memory_in = self._current_memory
            action, memory = self.call(obs, is_training=False)
            action = action.numpy()
        else:
            sess = self._get_session()
            if self._current_memory is None:
                self._current_memory = np.zeros((obs.shape[0], self.lstm_cell_size), dtype=np.float32), \
                    np.zeros((obs.shape[0], self.lstm_cell_size), dtype=np.float32)
            action, memory = sess.run(
                [self._action, self._memory_state], feed_dict={self._obs: obs, self._memory_in: self._current_memory})
        self._current_memory = memory
        return action[0]  # remove batch dimension

    def clear_memory(self) -> None:
        self._current_memory = None
