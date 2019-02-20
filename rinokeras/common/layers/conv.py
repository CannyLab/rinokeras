from typing import Optional
from tensorflow.keras.layers import Activation, Conv1D, Conv2D, Conv3D, Dropout

from .stack import Stack
from .normalization import LayerNorm
from .residual import Residual


class NormedConv(Stack):

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
                 layer_norm: bool = False,
                 activation: str = 'relu',
                 dropout: Optional[float] = None,
                 **kwargs) -> None:
        layer = [NormedConv(dimension, filters, kernel_size, layer_norm, activation) for _ in range(2)]
        if dropout is not None:
            layer.append(Dropout(dropout))
        super().__init__(Stack(layer), **kwargs)
