from tensorflow.keras import Model
from tensorflow.keras.layers import LSTMCell

import gym

from .RecurrentPolicy import RecurrentPolicy


class LSTMPolicy(RecurrentPolicy):

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
                 normalize_observations: bool = False,
                 use_rmc: bool = False,
                 **kwargs) -> None:

        recurrent_cell = LSTMCell(lstm_cell_size, implementation=2)
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
