from typing import Optional, Sequence, Type

import tensorflow as tf
from ray.rllib.models.lstm import add_time_dimension

from .StandardPolicy import StandardPolicy


class RecurrentPolicy(StandardPolicy):

    def __init__(self,
                 rnn_type: Type[tf.keras.layers.RNN],
                 num_outputs: int,
                 fcnet_hiddens: Sequence[int],
                 fcnet_activation: str,
                 conv_filters: Optional[Sequence[int]] = None,
                 conv_activation: str = 'relu',
                 lstm_cell_size: int = 256,
                 lstm_use_prev_action_reward: bool = False,
                 **options):
        super().__init__(
            num_outputs, fcnet_hiddens, fcnet_activation,
            conv_filters, conv_activation, **options)

        self._recurrent = True

        self._lstm_cell_size = lstm_cell_size
        self._lstm_use_prev_action_reward = lstm_use_prev_action_reward

        self.rnn = rnn_type(lstm_cell_size, return_state=True, return_sequences=True)

    def embed_features(self, inputs, seqlens, initial_state):
        outputs = super().embed_features(inputs, seqlens, initial_state)
        latent = outputs['latent']
        latent = add_time_dimension(latent, seqlens)
        latent, *lstm_state = self.rnn(latent, initial_state=initial_state)
        latent = tf.reshape(latent, [-1, latent.shape[-1]])
        state_out = list(lstm_state)
        outputs['latent'] = latent
        outputs['state_out'] = state_out
        return outputs

    def get_initial_state(self, inputs):
        return self.rnn.get_initial_state(inputs)

    @property
    def state_size(self) -> Sequence[int]:
        return self.rnn.cell.state_size
