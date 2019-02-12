from typing import Tuple, Optional, Dict

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Input
import tensorflow.keras.backend as K
import numpy as np

import gym

from rinokeras.common.layers import MaskedLSTM
from .StandardPolicy import StandardPolicy


class MemoryEmbedding(Model):

    def __init__(self, embedding_model: Model, lstm_cell_size: int = 256):
        super().__init__()
        self.embedding_model = embedding_model
        self.lstm_cell_size = lstm_cell_size
        self.cell = MaskedLSTM(lstm_cell_size, return_sequences=True, return_state=True)
        self.S = tf.placeholder(tf.float32, [None, 2 * lstm_cell_size])
        self.M = tf.placeholder(tf.float32, [None])
        self.initial_state = np.zeros([12, 2 * self.lstm_cell_size], dtype=np.float32)

    def call(self, obs, initial_state=None):
        embedding = self.embedding_model(obs)
        embedding.shape.assert_has_rank(2)
        # printop = tf.print(tf.shape(self.M), tf.shape(embedding), tf.shape(self.S))
        batch_size = tf.shape(self.S)[0]
        # with tf.control_dependencies([printop]):
        embedding = self.batch_to_seq(batch_size, -1, embedding)
        mask = self.batch_to_seq(batch_size, -1, self.M)

        memory_state = tf.split(self.S, axis=-1, num_or_size_splits=len(self.cell.state_size))

        embed_and_mask = tf.concat((embedding, mask[..., None]), -1)

        memory, *memory_state = self.cell(
            embed_and_mask, initial_state=memory_state)

        self.state = tf.concat(memory_state, -1)

        memory = self.seq_to_batch(memory)

        return memory

    def batch_to_seq(self, batch_size, seqlen, inputs):
        remaining_shape = inputs.shape[1:].as_list()
        outputs = tf.reshape(inputs, [batch_size, seqlen] + remaining_shape)
        return outputs

    def seq_to_batch(self, inputs):
        remaining_shape = inputs.shape[2:].as_list()
        outputs = tf.reshape(inputs, [-1] + remaining_shape)
        return outputs


class LSTMPolicy(StandardPolicy):

    def __init__(self,
                 obs_space: gym.Space,
                 act_space: gym.Space,
                 embedding_model: Model,
                 model_dim: int = 64,
                 n_layers_logits: int = 1,
                 n_layers_value: int = 1,
                 lstm_cell_size: int = 256,
                 take_greedy_actions: bool = False,
                 initial_logstd: float = 0,
                 extra_tensors: Optional[Dict[str, tf.Tensor]] = None,
                 normalize_observations: bool = False,
                 **kwargs) -> None:

        embedding_model = MemoryEmbedding(embedding_model, lstm_cell_size)
        super().__init__(
            obs_space, act_space, embedding_model,
            model_dim=model_dim,
            n_layers_logits=n_layers_logits,
            n_layers_value=n_layers_value,
            take_greedy_actions=take_greedy_actions,
            initial_logstd=initial_logstd,
            extra_tensors=extra_tensors,
            normalize_observations=normalize_observations,
            **kwargs)

    def call(self, obs, training=False):
        res = super().call(obs, training)
        self.S = self.embedding_model.S
        self.M = self.embedding_model.M
        self.state = self.embedding_model.state
        self.initial_state = self.embedding_model.initial_state
        return res

    # def _setup_memory_function(self):
        # return LSTM(self.lstm_cell_size, return_sequences=True, return_state=True)

    # def call(self, obs, training=False):
        # self._obs = obs
        # self.X = obs
#
        # bs, seqlen = tf.shape(obs)[0], tf.shape(obs)[1]
        # remaining_shape = obs.shape[2:].as_list()
        # obs = tf.reshape(obs, [bs * seqlen] + remaining_shape)
#
        # embedding = self.embedding_model(obs)
        # embedding.shape.assert_has_rank(2)
#
        # embedding = tf.reshape(embedding, (bs, seqlen, embedding.shape[-1]))
        # memory_out, memory_h, memory_c = self.memory_function(
            # embedding, initial_state=K.in_train_phase(None, self._memory_in, training=training))
        # memory_out = tf.reshape(memory_out, (bs * seqlen, memory_out.shape[-1]))
#
        # logits = self.logits_function(memory_out)
#
        # value = tf.squeeze(self.value_function(memory_out), -1)
        # action = self.action_distribution(logits, greedy=self.take_greedy_actions)
#
        # value = tf.reshape(value, (bs, seqlen))
        # logits = tf.reshape(logits, (bs, seqlen) + self.action_shape)
#
        # if training:
            # return logits, value
        # else:
            # return action, memory_h, memory_c

    # def clear_memory(self) -> None:
        # self._current_memory = None
