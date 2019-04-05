from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import VarianceScaling

from rinokeras.common.layers import DenseStack, Conv2DStack


class StandardPolicy(tf.keras.Model):

    def __init__(self,
                 num_outputs: int,
                 fcnet_hiddens: Sequence[int],
                 fcnet_activation: str,
                 conv_filters: Optional[Sequence[int]] = None,
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
            self.conv_layer = Conv2DStack(
                conv_filters, options['conv_kernel_size'], options['conv_strides'],
                activation=conv_activation, flatten_output=True)

        self.dense_layer = DenseStack(
            fcnet_hiddens,
            kernel_initializer=VarianceScaling(1.0),
            activation=fcnet_activation,
            output_activation=fcnet_activation)
        self.output_layer = Dense(
            num_outputs,
            kernel_initializer=VarianceScaling(0.01))

    def embed_features(self, inputs, seqlens, initial_state):
        features = inputs['obs']

        if self._use_conv:
            features = self.conv_layer(features)

        latent = self.dense_layer(features)

        outputs = {'latent': latent}
        return outputs

    def call(self, inputs, seqlens=None, initial_state=None):
        output = self.embed_features(inputs, seqlens, initial_state)
        logits = self.output_layer(output['latent'])
        output['logits'] = logits
        return output

    @property
    def recurrent(self) -> bool:
        return self._recurrent
