import math
from typing import Optional

from tensorflow.keras import Model
from tensorflow.keras.initializers import Orthogonal
from rinokeras.models.rmc import RelationalMemoryCoreCell
from rinokeras.common.layers import WeightNormDense as Dense

import gym

from .RecurrentPolicy import RecurrentPolicy


class RMCPolicy(RecurrentPolicy):

    def __init__(self,
                 obs_space: gym.Space,
                 act_space: gym.Space,
                 embedding_model: Model,
                 model_dim: int = 64,
                 n_layers_logits: int = 1,
                 n_layers_value: int = 1,
                 mem_slots: int = 10,
                 mem_size: int = 64,
                 n_heads: int = 1,
                 key_size: Optional[int] = None,
                 gate_style: str = 'unit',
                 treat_input_as_sequence: bool = False,
                 use_cross_attention: bool = False,
                 take_greedy_actions: bool = False,
                 initial_logstd: float = 0,
                 normalize_observations: bool = False,
                 use_rmc: bool = False,
                 **kwargs) -> None:

        recurrent_cell = RelationalMemoryCoreCell(
            mem_slots=mem_slots,
            mem_size=mem_size,
            n_heads=n_heads,
            key_size=key_size,
            dropout=None,
            gate_style=gate_style,
            treat_input_as_sequence=treat_input_as_sequence,
            use_cross_attention=use_cross_attention,
            return_attention_weights=False,
            kernel_initializer=Orthogonal(math.sqrt(2.0)))
        super().__init__(
            obs_space,
            act_space,
            embedding_model,
            recurrent_cell,
            model_dim=model_dim,
            n_layers_logits=n_layers_logits,
            n_layers_value=n_layers_value,
            take_greedy_actions=take_greedy_actions,
            initial_logstd=initial_logstd,
            normalize_observations=normalize_observations,
            **kwargs)
        # self.output_dense = Dense(512, activation='relu', kernel_initializer=Orthogonal(math.sqrt(2.0)))

        # def unroll_recurrance(self, embedding, mask, initial_state, nenv, nsteps):
            # output, state = super().unroll_recurrance(self, embedding, mask, initial_state, nenv, nsteps)
            # output = self.output_dense(output)
            # return output, state
