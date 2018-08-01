import collections
from typing import Optional, Sequence, Any, Union, Callable

import tensorflow as tf
from tensorflow.python.keras import backend as K  # pylint: disable=E0611


class RandomNoise(tf.keras.layers.Layer):
    """
    Adds gaussian random noise to input with trainable standard deviation.
    """

    def __init__(self, shape: Sequence[int], initial: float) -> None:
        super().__init__()
        self._shape = shape
        self._logstd = self.add_variable('logstd', shape, dtype=tf.float32,
                                         initializer=tf.constant_initializer(initial))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        epsilon = tf.random_normal(self._shape)
        return inputs + epsilon * tf.exp(self._logstd)

    @property
    def logstd(self) -> tf.Tensor:
        return self._logstd

    @property
    def std(self) -> tf.Tensor:
        return tf.exp(self._logstd)


# https://github.com/keras-team/keras/issues/3878
class LayerNorm(tf.keras.layers.Layer):
    """
    Does layer normalization from https://arxiv.org/abs/1607.06450.
    """

    def __init__(self, axis: Union[Sequence[int], int] = -1, eps: float = 1e-6, **kwargs) -> None:
        if isinstance(axis, collections.Sequence):
            self.axis: Sequence[int] = axis
        else:
            self.axis: Sequence[int] = (axis,)
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape: Sequence[tf.Dimension]) -> None:
        shape = [input_shape[axis] for axis in self.axis]

        self.gamma = self.add_variable(name='gamma',
                                       shape=shape,
                                       initializer=tf.keras.initializers.Ones(),
                                       trainable=True)
        self.beta = self.add_variable(name='beta',
                                      shape=shape,
                                      initializer=tf.keras.initializers.Zeros(),
                                      trainable=True)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        std = K.std(inputs, axis=self.axis, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

# I presume this is just how Sequential is added but at the moment Sequential
# requires input size to be specified at the begining


class Stack(tf.keras.Model):
    """
    A re-implementation of Keras's Sequential layer to work well with tf eager.
    """
    def __init__(self, layers: Optional[Sequence[Any]] = None) -> None:
        super().__init__()
        self._call = None
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Callable[[tf.Tensor], tf.Tensor]) -> None:
        self._layers.append(layer)

    def call(self, inputs, **kwargs):
        output = inputs
        for layer in self._layers:
            output = layer(output, **kwargs)
        return output


class Conv2DStack(Stack):
    """
    A stack of convolutional layers. Can optionally do batch normalization after each layer.
    """
    def __init__(self, 
                 layers: Sequence[tuple], 
                 batch_norm: bool = False, 
                 activation: str = 'relu', 
                 padding: str = 'same', 
                 flatten_output: bool = True) -> None:
        super().__init__()
        if layers is None:
            layers = []
        for layer in layers:
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(tf.keras.layers.Conv2D(*layer, padding=padding))
            if batch_norm:
                self.add(tf.keras.layers.BatchNormalization())
            self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.Flatten())


class DenseStack(Stack):
    """
    A stack of fully connected layers. Can do batch norm and specify an alternate output activation.
    """
    def __init__(self, 
                 layers: Sequence[Union[tuple, int]], 
                 batch_norm: bool = False, 
                 activation: str = 'relu', 
                 output_activation: Optional[str] = None) -> None:
        super().__init__()
        if layers is None:
            layers = []
        for _, layer in enumerate(layers[:-1]):
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(tf.keras.layers.Dense(*layer))
            if batch_norm:
                self.add(tf.keras.layers.BatchNormalization())
            self.add(tf.keras.layers.Activation(activation))

        out_layer = layers[-1]
        if not isinstance(out_layer, collections.Iterable):
            out_layer = (out_layer,)
        self.add(tf.keras.layers.Dense(*out_layer))
        if output_activation is not None:
            self.add(tf.keras.layers.Activation(output_activation))


class Residual(tf.keras.Model):
    """
    Adds a residual connection between layers. If input to layer is a tuple, adds output to the first element
    of the tuple.
    """
    def __init__(self, layer: Callable) -> None:
        super().__init__()
        self.layer = layer

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        layer_out = self.layer(inputs, *args, **kwargs)
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        residual = inputs + layer_out

        return residual


class Highway(tf.keras.Model):
    """
    Implementation of a highway layer. Can use convolutional or fully connected layer. 

    From the paper: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, 
                 convolution: bool = False,
                 activation: str = 'relu',
                 gate_bias: float = -3.0,
                 dropout: Optional[float] = None) -> None:
        super().__init__()
        self._convolution = convolution
        self.activation = activation
        self._gate_initializer = tf.keras.initializers.Constant(gate_bias)
        self.dropout = None if dropout is None else tf.keras.layers.Dropout(
            dropout)

    def build(self, input_shape: Sequence[tf.Dimension]) -> None:
        units = input_shape[-1]
        if self._convolution:
            self.gate = tf.keras.layers.Conv1D(filters=units,
                                               kernel_size=1,
                                               padding='same',
                                               acitvation='sigmoid',
                                               use_bias=True,
                                               bias_initializer=self._gate_initializer)
            self.layer = tf.keras.layers.Conv1D(filters=units,
                                                kernel_size=1,
                                                padding='same',
                                                activation=self.activation)
        else:
            self.gate = tf.keras.layers.Dense(units=units,
                                              activation='sigmoid',
                                              use_bias=True,
                                              bias_initializer=self._gate_initializer)
            self.layer = tf.keras.layers.Dense(units=units,
                                               activation=self.activation)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        gated = self.gate(inputs)
        transformed = self.layer(inputs)
        if self.dropout:
            transformed = self.dropout(transformed)
        return gated * transformed + (1 - gated) * inputs


class PositionEmbedding(tf.keras.Model):
    """
    Adds positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self):
        super().__init__()

    def build(self, input_shape: Sequence[tf.Dimension]) -> None:
        hidden_size = input_shape[-1]
        assert hidden_size % 2 == 0, 'Model vector size must be even for sinusoidal encoding'
        power = tf.range(0, hidden_size.value, 2,
                         dtype=tf.float32) / hidden_size.value
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        assert inputs.shape[-1] == self.hidden_size, 'Input final dim must match model hidden size'

        sequence_length = tf.shape(inputs)[1]
        seq_pos = tf.cast(tf.range(1, sequence_length + 1)
                          [None, :], tf.float32)  # 1-index positions

        index = seq_pos[:, :, None] / self.divisor

        sin_embedding = tf.sin(index)
        cos_embedding = tf.cos(index)

        position_embedding = tf.stack((sin_embedding, cos_embedding), -1)
        position_shape = (1, sequence_length, self.hidden_size)

        position_embedding = tf.reshape(position_embedding, position_shape)

        return inputs + position_embedding


class PositionEmbedding2D(PositionEmbedding):
    """
    Adds a 2D positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self):
        super().__init__()

    def build(self, input_shape: Sequence[tf.Dimension]) -> None:
        hidden_size = input_shape[-1]
        assert hidden_size % 4 == 0, 'Model vector size must be multiple of four for 2D sinusoidal encoding'

        power = tf.range(0, self.hidden_size, 4,
                         dtype=tf.float32) / self.hidden_size
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, Width, Height, Channels]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, Width, Height, Channels]
        """
        width, height, channels = inputs.shape[1:]
        assert channels == self.hidden_size, 'Input final dim must match model hidden size'

        width_pos = tf.cast(tf.range(1, width + 1)[None, :], tf.float32)
        height_pos = tf.cast(tf.range(1, height + 1)[None, :], tf.float32)

        width_embed = width_pos[:, :, None] / self.divisor
        height_embed = height_pos[:, :, None] / self.divisor

        width_embed = tf.tile(width_embed[:, :, None, :], (1, 1, height, 1))
        height_embed = tf.tile(height_embed[:, None, :, :], (1, width, 1, 1))

        width_sin_embed = tf.sin(width_embed)
        width_cos_embed = tf.cos(width_embed)
        height_sin_embed = tf.sin(height_embed)
        height_cos_embed = tf.cos(height_embed)

        position_embedding = tf.stack((width_sin_embed, width_cos_embed,
                                       height_sin_embed, height_cos_embed), -1)
        position_embedding = tf.reshape(
            position_embedding, (1, width, height, self.hidden_size))

        return inputs + position_embedding
