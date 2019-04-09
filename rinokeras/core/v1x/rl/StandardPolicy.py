from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import Dense
from ray.rllib.models.misc import normc_initializer

from rinokeras.layers import DenseStack, Conv2DStack


ConvLayerSpec = Tuple[int, Union[int, Tuple[int, int]], int]


class StandardPolicy(tf.keras.Model):

    def __init__(self,
                 num_outputs: int,
                 fcnet_hiddens: Sequence[int],
                 fcnet_activation: str,
                 conv_filters: Optional[Sequence[ConvLayerSpec]] = None,
                 conv_activation: str = 'relu',
                 **options):
        super().__init__()

        self._num_outputs = num_outputs
        self._fcnet_hiddens = fcnet_hiddens
        self._fcnet_activation = fcnet_activation
        self._use_conv = conv_filters is not None
        self._conv_filters = conv_filters
        self._conv_activation = conv_activation
        self._recurrent = False
        self._options = options

        if conv_filters is not None:
            filters, kernel_size, strides = list(zip(*conv_filters))
            self.conv_layer = Conv2DStack(
                filters, kernel_size, strides,
                padding='valid',
                activation=conv_activation,
                flatten_output=True)

        self.dense_layer = DenseStack(
            fcnet_hiddens,
            kernel_initializer=normc_initializer(1.0),
            activation=fcnet_activation,
            output_activation=fcnet_activation)

        # WARNING: DO NOT CHANGE KERNEL INITIALIZER!!!
        # PPO/Gradient based methods are extremely senstive to this and will break
        # Don't alter this unless you're sure you know what you're doing.
        self.output_layer = Dense(
            num_outputs,
            kernel_initializer=normc_initializer(0.01))

    def call(self, inputs, seqlens=None, initial_state=None):
        features = inputs['obs']

        if self._use_conv:
            features = self.conv_layer(features)

        latent = self.dense_layer(features)
        logits = self.output_layer(latent)

        output = {'latent': latent, 'logits': logits}

        return output

    @property
    def recurrent(self) -> bool:
        return self._recurrent
