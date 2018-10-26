import collections
from typing import Optional, Sequence, Any, Union, Callable

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Conv2D, Dropout, Conv2DTranspose, \
    BatchNormalization, Flatten, Activation
import tensorflow.keras.backend as K  # pylint: disable=E0611


class RandomNoise(Layer):
    """
    Adds gaussian random noise to input with trainable standard deviation.
    """

    def __init__(self, shape: Sequence[int], initial: float) -> None:
        super(RandomNoise, self).__init__()
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
class LayerNorm(Layer):
    """
    Does layer normalization from https://arxiv.org/abs/1607.06450.
    """

    def __init__(self, axis: Union[Sequence[int], int] = -1, eps: float = 1e-6, **kwargs) -> None:
        if isinstance(axis, collections.Sequence):
            self.axis: Sequence[int] = axis
        else:
            self.axis: Sequence[int] = (axis,)
        self.eps = eps
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
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

    def call(self, inputs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        std = K.std(inputs, axis=self.axis, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

# I presume this is just how Sequential is added but at the moment Sequential
# requires input size to be specified at the begining


class Stack(Model):
    """
    A re-implementation of Keras's Sequential layer to work well with tf eager.
    """
    def __init__(self, layers: Optional[Sequence[Any]] = None, *args, **kwargs) -> None:
        super(Stack, self).__init__(*args, **kwargs)
        self._call = None
        self._layer_list = tf.contrib.checkpoint.List()
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
        super(Conv2DStack, self).__init__()
        assert len(filters) == len(kernel_size) == len(strides), 'Filters, kernels, and strides must have same length'
        for fsize, ks, stride in zip(filters, kernel_size, strides):
            self.add(Conv2D(fsize, ks, stride, padding=padding, **kwargs))
            if batch_norm:
                self.add(BatchNormalization())
            self.add(Activation(activation))
        if flatten_output:
            self.add(Flatten())


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
        super(Deconv2DStack, self).__init__()
        assert len(filters) == len(kernel_size) == len(strides), 'Filters, kernels, and strides must have same length'
        for fsize, ks, stride in zip(filters, kernel_size, strides):
            self.add(Conv2DTranspose(fsize, ks, stride, padding=padding, **kwargs))
            if batch_norm:
                self.add(BatchNormalization())
            self.add(Activation(activation))
        if flatten_output:
            self.add(Flatten())


class DenseStack(Stack):
    """
    A stack of fully connected layers. Can do batch norm and specify an alternate output activation.
    """
    def __init__(self,
                 layers: Sequence[Union[tuple, int]],
                 batch_norm: bool = False,
                 activation: str = 'relu',
                 output_activation: Optional[str] = None,
                 **kwargs) -> None:
        super(DenseStack, self).__init__()
        if layers is None:
            layers = []
        for _, layer in enumerate(layers[:-1]):
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(Dense(*layer, **kwargs))
            if batch_norm:
                self.add(BatchNormalization())
            self.add(Activation(activation))

        out_layer = layers[-1]
        if not isinstance(out_layer, collections.Iterable):
            out_layer = (out_layer,)
        self.add(Dense(*out_layer, **kwargs))
        if output_activation is not None:
            self.add(Activation(output_activation))


class DenseTranspose(Layer):
    """Multiply by the transpose of a dense layer
    """
    def __init__(self, other_layer):
        super(DenseTranspose, self).__init__()
        self.other_layer = other_layer

    def call(self, x):
        return K.dot(x - K.stop_gradient(self.other_layer.b), K.transpose(K.stop_gradient(self.other_layer.W)))


class Residual(Model):
    """
    Adds a residual connection between layers. If input to layer is a tuple, adds output to the first element
    of the tuple.
    """
    def __init__(self, layer: Callable) -> None:
        super(Residual, self).__init__()
        self.layer = layer

    def call(self, inputs, *args, **kwargs):
        layer_out = self.layer(inputs, *args, **kwargs)
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        residual = inputs + layer_out

        return residual


class LayerDropout(Model):
    """
    Optionally drops a full layer. Output is x with probability rate and f(x) with probability (1 - rate).

    Args:
        layer_call (Callable[[], Any]): Function that returns output of layer on inputs
        inputs (Any): What to return if the layer is dropped
        rate (float): Rate at which to drop layers

    Returns:
        Any: Either inputs or output of layer_call function.
    """

    def __init__(self, rate: float, *args, **kwargs) -> None:
        super(LayerDropout, self).__init__(*args, **kwargs)
        self.rate = rate

    def call(self, layer_output, inputs, training=False):
        if training:
            return K.switch(K.random_uniform([]) > self.rate, layer_output, inputs)
        else:
            return layer_output


class MaskInput(Layer):
    """
    Replaces some percentage of the input with a mask token. Used for implementing BERT style models.

    Based on https://arxiv.org/abs/1810.04805.

    Args:
        percentage (float): Percentage of input tokens to mask
        mask_token (int): Token to replace masked input with
    """

    def __init__(self, percentage: float, mask_token: int, *args, **kwargs) -> None:
        super(MaskInput, self).__init__(*args, **kwargs)
        if not 0 <= percentage < 1:
            raise ValueError("Masking percentage must be in [0, 1). Received {}".format(percentage))
        self.percentage = percentage
        self.mask_token = mask_token

    def call(self, inputs: tf.Tensor, valid_mask: Optional[tf.Tensor]=None):
        """
        Args:
            inputs (tf.Tensor[ndims=2, int]): Tensor of values to mask
            valid_mask (Optional[tf.Tensor[bool]]): Locations in the inputs to that are valid
                                                     (i.e. not padding, start tokens, etc.)
        Returns:
            masked_inputs (tf.Tensor[ndims=2, int]): Tensor of masked values
            bert_mask: Locations in the input that were masked
        """

        bert_mask = K.random_uniform(K.shape(inputs)) < self.percentage

        if valid_mask is not None:
            bert_mask &= valid_mask

        masked_inputs = inputs * tf.cast(~bert_mask, inputs.dtype)  # type: ignore
        masked_inputs = masked_inputs + self.mask_token * tf.cast(bert_mask, inputs.dtype)  # type: ignore

        return masked_inputs, bert_mask


class Highway(Model):
    """
    Implementation of a highway layer. Can use convolutional or fully connected layer.

    From the paper: https://arxiv.org/abs/1607.06450
    """
    def __init__(self,
                 convolution: bool = False,
                 activation: str = 'relu',
                 gate_bias: float = -3.0,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(Highway, self).__init__()
        self._convolution = convolution
        self.activation = activation
        self._gate_initializer = tf.keras.initializers.Constant(gate_bias)
        self.dropout_weight = 0 if dropout is None else dropout
        self.dropout = Dropout(self.dropout_weight)

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

    def build(self, input_shape):
        units = input_shape[-1]
        if self._convolution:
            self.gate = Conv1D(filters=units,
                               kernel_size=1,
                               padding='same',
                               activation='sigmoid',
                               use_bias=True,
                               bias_initializer=self._gate_initializer,
                               kernel_regularizer=self.kernel_regularizer,
                               bias_regularizer=self.bias_regularizer,
                               activity_regularizer=self.activity_regularizer)
            self.layer = Conv1D(filters=units,
                                kernel_size=1,
                                padding='same',
                                activation=self.activation,
                                kernel_regularizer=self.kernel_regularizer,
                                bias_regularizer=self.bias_regularizer,
                                activity_regularizer=self.activity_regularizer)
        else:
            self.gate = Dense(units=units,
                              activation='sigmoid',
                              use_bias=True,
                              bias_initializer=self._gate_initializer,
                              kernel_regularizer=self.kernel_regularizer,
                              bias_regularizer=self.bias_regularizer,
                              activity_regularizer=self.activity_regularizer)
            self.layer = Dense(units=units,
                               activation=self.activation,
                               kernel_regularizer=self.kernel_regularizer,
                               bias_regularizer=self.bias_regularizer,
                               activity_regularizer=self.activity_regularizer)

    def call(self, inputs):
        gated = self.gate(inputs)
        transformed = self.layer(inputs)
        if self.dropout:
            transformed = self.dropout(transformed)
        return gated * transformed + (1 - gated) * inputs


class PositionEmbedding(Model):
    """
    Adds positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self) -> None:
        super(PositionEmbedding, self).__init__()

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        assert hidden_size % 2 == 0, 'Model vector size must be even for sinusoidal encoding'
        power = tf.range(0, hidden_size.value, 2,
                         dtype=tf.float32) / hidden_size.value
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def call(self, inputs):
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
        super(PositionEmbedding2D, self).__init__()

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        assert hidden_size % 4 == 0, 'Model vector size must be multiple of four for 2D sinusoidal encoding'

        power = tf.range(0, hidden_size.value, 4,
                         dtype=tf.float32) / hidden_size.value
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def call(self, inputs):
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


__all__ = ['RandomNoise', 'LayerNorm', 'Stack', 'Conv2DStack', 'DenseStack', 'DenseTranspose',
           'Residual', 'Highway', 'PositionEmbedding', 'PositionEmbedding2D', 'MaskInput']
