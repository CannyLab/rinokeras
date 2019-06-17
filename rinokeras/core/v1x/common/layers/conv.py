from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv1D, Conv2D, Conv3D, Dropout, BatchNormalization, Layer, LeakyReLU

from rinokeras.core.v1x.common.layers.stack import Stack
from rinokeras.core.v1x.common.layers.normalization import LayerNorm
from rinokeras.core.v1x.common.layers.residual import Residual


class NormedConvStack(Stack):

    def __init__(self,
                 dimension: int,
                 filters: int,
                 kernel_size: int,
                 layer_norm: bool = False,
                 activation: str = 'relu') -> None:
        super().__init__()
        assert 1 <= dimension <= 3
        if layer_norm:
            self.add(LayerNorm())
        self.add(Activation(activation))

        conv_func = [Conv1D, Conv2D, Conv3D]
        self.add(conv_func[dimension - 1](
            filters=filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=True))

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            if mask.shape.ndims == 2:
                mask = mask[:, :, None]
            inputs = inputs * mask
        return super().call(inputs, **kwargs)


class PaddedConv(Stack):

    def __init__(self,
                 dimension: int,
                 filters: int,
                 kernel_size: int,
                 dilation_rate: int = 1,
                 activation: str = 'relu',
                 dropout: Optional[float] = None) -> None:
        super().__init__()
        assert 1 <= dimension <= 3
        conv_func = [Conv1D, Conv2D, Conv3D]

        def get_activation():
            if activation == 'glu':
                return GLUActivation()
            elif activation == 'lrelu':
                return LeakyReLU()
            else:
                return Activation(activation)

        self.add(BatchNormalization())
        self.add(get_activation())
        self.add(conv_func[dimension - 1](
            filters=filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=True,
            activation='linear', dilation_rate=dilation_rate, kernel_initializer='he_normal'))
        if dropout is not None:
            self.add(Dropout(dropout))

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            if mask.shape.ndims == inputs.shape.ndims - 1:
                mask = tf.expand_dims(mask, -1)
            inputs = inputs * mask
        return super().call(inputs, mask=mask)


class GLUActivation(Layer):

    def call(self, inputs):
        output, gate = tf.split(inputs, axis=-1, num_or_size_splits=2)
        return output * tf.nn.sigmoid(gate)


class ResidualBlock(Residual):

    def __init__(self,
                 dimension: int,
                 filters: int,
                 kernel_size: int,
                 activation: str = 'relu',
                 dilation_rate: int = 1,
                 layer_norm: bool = False,
                 dropout: Optional[float] = None,
                 add_checkpoint: bool = False,  # used with memory saving gradients
                 **kwargs) -> None:

        self._add_checkpoint = add_checkpoint
        layer = Stack()
        if layer_norm:
            layer.add(LayerNorm())
        layer.add(PaddedConv(dimension, filters, kernel_size, dilation_rate, activation, dropout))
        layer.add(PaddedConv(dimension, filters, kernel_size, dilation_rate, activation, dropout))

        super().__init__(layer, **kwargs)

    def call(self, inputs, *args, **kwargs):
        output = super().call(inputs, *args, **kwargs)

        if self._add_checkpoint:
            tf.add_to_collection('checkpoints', output)

        return output


class GroupedConvolution(tf.keras.Model):
    def __init__(self, cardinality: int = 1, n_filters: int = 64, kernel_size: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (1,1)) -> None:
        super(GroupedConvolution, self).__init__()
        self.cardinality = cardinality

        if self.cardinality == 1:
            self.output_layer = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, padding='same')
        else:
            if (n_filters % self.cardinality != 0):
                raise ValueError('Residual grouped convolution filters must be divisible by the cardinality')

            self._dim = n_filters // self.cardinality

            self._layer_list = tf.contrib.checkpoint.List()
            for idx in range(self.cardinality):
                group = tf.keras.layers.Lambda(lambda z: z[:,:,:, idx * self._dim: (idx + 1) * self._dim])
                group = tf.keras.layers.Conv2D(filters=self._dim, kernel_size=kernel_size, strides=stride, padding='same')
                self._layer_list.append(group)

    def call(self, inputs, *args, **kwargs):
        if self.cardinality == 1:
            return self.output_layer(inputs)
        else:
            layers = [layer(inputs) for layer in self._layer_list]
            return tf.keras.layers.Concatenate()(layers)
