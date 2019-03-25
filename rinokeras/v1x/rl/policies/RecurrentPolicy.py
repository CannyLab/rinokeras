import collections

import tensorflow as tf
from tensorflow.keras import Model
import numpy as np

import gym

from rinokeras.utils import get_shape
from .StandardPolicy import StandardPolicy


class RecurrentPolicy(StandardPolicy):

    def __init__(self,
                 obs_space: gym.Space,
                 act_space: gym.Space,
                 embedding_model: Model,
                 recurrent_cell: Model,
                 model_dim: int = 64,
                 n_layers_logits: int = 1,
                 n_layers_value: int = 1,
                 take_greedy_actions: bool = False,
                 initial_logstd: float = 0,
                 normalize_observations: bool = False,
                 use_rmc: bool = False,
                 **kwargs) -> None:

        super().__init__(
            obs_space, act_space, embedding_model,
            model_dim=model_dim,
            n_layers_logits=n_layers_logits,
            n_layers_value=n_layers_value,
            take_greedy_actions=take_greedy_actions,
            initial_logstd=initial_logstd,
            normalize_observations=normalize_observations,
            **kwargs)
        self.cell = recurrent_cell

    def mask_state(self, inputs, state, mask):
        initial_state = self.cell.get_initial_state(inputs)
        state = [s * (1 - mask) + init_s * mask for s, init_s in zip(state, initial_state)]
        return state

    def unroll_recurrance(self, embedding, mask, initial_state, nenv, nsteps):
        unrolled_embedding = self.batch_to_seq(embedding, nenv, nsteps)
        unrolled_masks = self.batch_to_seq(mask[..., None], nenv, nsteps)

        state_size = self.cell.state_size
        if not isinstance(state_size, collections.Iterable):
            state_size = (state_size,)

        state = tf.split(initial_state, axis=-1, num_or_size_splits=state_size)
        unrolled_outputs = []
        for embed_t, mask_t in zip(unrolled_embedding, unrolled_masks):
            state = self.mask_state(embed_t, state, mask_t)
            output, state = self.cell(embed_t, state)
            unrolled_outputs.append(output)

        state = tf.concat(state, -1)
        outputs = self.seq_to_batch(unrolled_outputs)

        return outputs, state

    def call(self, obs, mask, initial_state, nenv, nsteps, training=False):
        self._obs = obs
        self._initial_state = initial_state
        self._mask = mask

        if self.normalize_observations and obs.dtype == tf.float32:
            obs = self.batch_norm(obs)
            obs = tf.clip_by_value(obs, -5.0, 5.0)

        embedding = self.embedding_model(obs)
        embedding, state = self.unroll_recurrance(embedding, mask, initial_state, nenv, nsteps)
        # embedding_list = self.batch_to_seq(nenv, nsteps, embedding)
        # mask_list = self.batch_to_seq(nenv, nsteps, mask[..., None])
#
        # h, c = tf.split(initial_state, axis=-1, num_or_size_splits=2)
        # embedding_out = []
        # for embed_t, mask_t in zip(embedding_list, mask_list):
            # h = h * (1 - mask_t)
            # c = c * (1 - mask_t)
            # _, (h, c) = self.cell(embed_t, (h, c))
            # embedding_out.append(h)
#
        # state = tf.concat((h, c), -1)
#
        # embedding = self.seq_to_batch(embedding_out)

        logits = self.logits_function(embedding)

        value = tf.squeeze(self.value_function(embedding), -1)

        action = self.pd(logits, greedy=self.take_greedy_actions)
        neglogpac = self.neglogp(action)
        initial_state_numpy = np.zeros([nenv, self.state_size], dtype=np.float32)

        return {'latent': embedding,
                'q': logits,
                'vf': value,
                'state': state,
                'action': action,
                'neglogp': neglogpac,
                'S': initial_state,
                'M': mask,
                'initial_state': initial_state_numpy}

    def batch_to_seq(self, inputs, batch_size, seqlen):
        remaining_shape = [get_shape(inputs, dim) for dim in range(1, inputs.shape.ndims)]
        outputs = tf.reshape(inputs, [batch_size, seqlen] + remaining_shape)
        return [tf.squeeze(output, [1]) for output in tf.split(outputs, axis=1, num_or_size_splits=seqlen)]

    def seq_to_batch(self, inputs):
        inputs = tf.stack(inputs, axis=1)
        remaining_shape = inputs.shape[2:].as_list()
        batch_size = get_shape(inputs, 0)
        seqlen = get_shape(inputs, 1)
        outputs = tf.reshape(inputs, [batch_size * seqlen] + remaining_shape)
        return outputs

    @property
    def state_size(self) -> int:
        return sum(self.cell.state_size) if \
            isinstance(self.cell.state_size, collections.Iterable) else self.cell.state_size
