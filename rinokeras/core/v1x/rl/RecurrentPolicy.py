from typing import Optional, Sequence, Type, Dict, Any

import tensorflow as tf
from ray.rllib.models.lstm import add_time_dimension

from .StandardPolicy import StandardPolicy, ConvLayerSpec


class RecurrentPolicy(StandardPolicy):

    def __init__(self,
                 rnn_type: Type[tf.keras.layers.RNN],
                 num_outputs: int,
                 fcnet_hiddens: Sequence[int],
                 fcnet_activation: str,
                 conv_filters: Optional[Sequence[ConvLayerSpec]] = None,
                 conv_activation: str = 'relu',
                 lstm_cell_size: int = 256,
                 lstm_use_prev_action_reward: bool = False,
                 recurrent_args: Optional[Dict[str, Any]] = None,
                 **options):
        super().__init__(
            num_outputs, fcnet_hiddens, fcnet_activation,
            conv_filters, conv_activation, **options)

        self._recurrent = True

        self._lstm_cell_size = lstm_cell_size
        self._lstm_use_prev_action_reward = lstm_use_prev_action_reward

        if recurrent_args is None:
            recurrent_args = {}

        self.rnn = rnn_type(lstm_cell_size, return_state=True, return_sequences=True, **recurrent_args)

    def call(self, inputs, seqlens=None, initial_state=None):
        features = inputs['obs']

        if self._use_conv:
            features = self.conv_layer(features)

        features = add_time_dimension(features, seqlens)
        latent, *rnn_state = self.rnn(features, initial_state=initial_state)
        latent = tf.reshape(latent, [-1, latent.shape[-1]])

        state_out = list(rnn_state)

        # latent = self.dense_layer(latent)
        logits = self.output_layer(latent)

        output = {'latent': latent, 'logits': logits, 'state_out': state_out}

        return output

    def get_initial_state(self, inputs):
        return self.rnn.get_initial_state(inputs)

    @property
    def state_size(self) -> Sequence[int]:
        return self.rnn.cell.state_size
