from typing import Sequence, Optional

from tensorflow.keras.layers import LSTM

from .RecurrentPolicy import RecurrentPolicy


class LSTMPolicy(RecurrentPolicy):

    def __init__(self,
                 num_outputs: int,
                 fcnet_hiddens: Sequence[int],
                 fcnet_activation: str,
                 conv_filters: Optional[Sequence[int]] = None,
                 conv_activation: str = 'relu',
                 lstm_cell_size: int = 256,
                 lstm_use_prev_action_reward: bool = False,
                 **options):
        super().__init__(
            LSTM, num_outputs, fcnet_hiddens, fcnet_activation,
            conv_filters, conv_activation, lstm_cell_size,
            lstm_use_prev_action_reward, **options)
