from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, Conv2D, Conv3D, Dropout

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
            filters=filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=not layer_norm))


class ResidualBlock(Residual):

    def __init__(self,
                 dimension: int,
                 filters: int,
                 kernel_size: int,
                 n_layers: int = 2,
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 dropout: Optional[float] = None,
                 **kwargs) -> None:
        layer = [NormedConvStack(dimension, filters, kernel_size, layer_norm, activation) for _ in range(n_layers)]
        if dropout is not None:
            layer.append(Dropout(dropout))
        super().__init__(Stack(layer), **kwargs)


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