from typing import Optional
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
