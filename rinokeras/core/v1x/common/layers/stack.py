"""

Stack-type layers
#TODO: At some point these should be replaced with Keras Sequential
"""

import collections
from typing import Optional, Dict, Sequence, Any, Union, List

from tensorflow.keras import Model  # pylint: disable=F0401
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
    BatchNormalization, Flatten, Activation, Dense, Layer  # pylint: disable=F0401

from .normalization import WeightNormDense
from .dropout import LayerDropout


class Stack(Model):
    """
    A re-implementation of Keras's Sequential layer to work well with tf eager.
    """
    def __init__(self, layers: Optional[Sequence[Any]] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._layer_list = []  # type: List[Layer]
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layer_list.append(layer)

    def call(self, inputs, **kwargs):
        output = inputs
        for layer in self._layer_list:
            output = layer(output, **kwargs)
        return output

    def get_config(self) -> Dict:
        config = {
            'layers': [layer.__class__.from_config(layer.get_config()) for layer in self._layers],
        }
        return config

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)


class LayerDropoutStack(Stack):
    def __init__(self, layers: Optional[Sequence[Any]] = None, layer_dropout: Optional[float] = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._layer_list = []  # type: List[Layer]
        self._layer_dropout_list = []
        self.layer_dropout = layer_dropout
        if layers is not None:
            for layer in layers:
                self.add(layer)


    def add(self, layer):
        self._layer_list.append(layer)
        self._layer_dropout_list.append(LayerDropout(self.layer_dropout))

    def call(self, inputs, **kwargs):
        output = inputs
        for idx, layer in enumerate(self._layer_list):
            output_ld = layer(output, **kwargs)
            output = self._layer_dropout_list[idx](output_ld,output)
        return output

    def get_config(self) -> Dict:
        config = {
            'layer_dropout': self.layer_dropout,
            'layers': [layer.__class__.from_config(layer.get_config()) for layer in self._layers],
        }
        return config

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)


class Conv2DStack(Stack):
    """
    A stack of convolutional layers. Can optionally do batch normalization after each layer.
    """
    def __init__(self,
                 filters: Sequence[int],
                 kernel_size: Sequence[int],
                 strides: Sequence[int],
                 batch_norm: bool = False,
                 activation: str = 'relu',
                 padding: str = 'same',
                 flatten_output: bool = True,
                 **kwargs) -> None:
        super().__init__()
        assert len(filters) == len(kernel_size) == len(strides), 'Filters, kernels, and strides must have same length'
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.batch_norm = batch_norm
        self.activation = activation
        self.padding = padding
        self.flatten_output = flatten_output

        for fsize, ks, stride in zip(filters, kernel_size, strides):
            self.add(Conv2D(fsize, ks, stride, padding=padding, **kwargs))
            if batch_norm:
                self.add(BatchNormalization())
            self.add(Activation(activation))
        if flatten_output:
            self.add(Flatten())

    def get_config(self) -> Dict:
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'batch_norm': self.batch_norm,
            'activation': self.activation,
            'padding': self.padding,
            'flatten_output': self.flatten_output
        }

        base_config = super().get_config()
        if 'layers' in base_config:
            del base_config['layers']
        return dict(list(base_config.items()) + list(config.items()))


class Deconv2DStack(Stack):
    """
    A stack of deconvolutional layers. Can optionally do batch normalization after each layer.
    Note:  Deconvolution in tf.keras perform transpose convolution, so if you want
    UPconvolution's stride to be 1/2, write 2 in this case
    """
    def __init__(self,
                 filters: Sequence[int],
                 kernel_size: Sequence[int],
                 strides: Sequence[int],
                 batch_norm: bool = False,
                 activation: str = 'relu',
                 padding: str = 'same',
                 flatten_output: bool = True,
                 **kwargs) -> None:
        super().__init__()
        assert len(filters) == len(kernel_size) == len(strides), 'Filters, kernels, and strides must have same length'
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.batch_norm = batch_norm
        self.activation = activation
        self.padding = padding
        self.flatten_output = flatten_output

        for fsize, ks, stride in zip(filters, kernel_size, strides):
            self.add(Conv2DTranspose(fsize, ks, stride, padding=padding, **kwargs))
            if batch_norm:
                self.add(BatchNormalization())
            self.add(Activation(activation))
        if flatten_output:
            self.add(Flatten())

    def get_config(self) -> Dict:
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'batch_norm': self.batch_norm,
            'activation': self.activation,
            'padding': self.padding,
            'flatten_output': self.flatten_output
        }

        base_config = super().get_config()
        if 'layers' in base_config:
            del base_config['layers']
        return dict(list(base_config.items()) + list(config.items()))


class DenseStack(Stack):
    """
    A stack of fully connected layers. Can do batch norm and specify an alternate output activation.
    """
    def __init__(self,
                 layers: Sequence[Union[tuple, int]],
                 batch_norm: bool = False,
                 activation: str = 'relu',
                 output_activation: Optional[str] = None,
                 use_weight_norm: bool = True,
                 **kwargs) -> None:
        super().__init__()

        self.initial_layer_config = tuple(layers)
        self.batch_norm = batch_norm
        self.activation = activation
        self.output_activation = output_activation

        if layers is None:
            layers = []
        for _, layer in enumerate(layers[:-1]):
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            if use_weight_norm:
                self.add(WeightNormDense(*layer, **kwargs))
            else:
                self.add(Dense(*layer, **kwargs))
            if batch_norm:
                self.add(BatchNormalization())
            self.add(Activation(activation))

        out_layer = layers[-1]
        if not isinstance(out_layer, collections.Iterable):
            out_layer = (out_layer,)
        if use_weight_norm:
            self.add(WeightNormDense(*out_layer, **kwargs))
        else:
            self.add(Dense(*out_layer, **kwargs))
        if output_activation is not None:
            self.add(Activation(output_activation))

    def get_config(self) -> Dict:
        config = {
            'layers': self.initial_layer_config,
            'batch_norm': self.batch_norm,
            'activation': self.activation,
            'output_activation': self.output_activation
        }

        base_config = super().get_config()
        if 'layers' in base_config:
            del base_config['layers']
        return dict(list(base_config.items()) + list(config.items()))
