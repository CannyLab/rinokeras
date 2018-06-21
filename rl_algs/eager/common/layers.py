import collections
import tensorflow as tf
from tensorflow.python.keras import backend as K

class RandomNoise(tf.keras.layers.Layer):

    def __init__(self, shape, initial):
        super().__init__()
        self._shape = shape
        self._logstd = self.add_variable('logstd', shape, dtype=tf.float32,
                                            initializer=tf.constant_initializer(initial))
    def call(self, inputs):
        epsilon = tf.random_normal(self._shape)
        return inputs + epsilon * tf.exp(self._logstd)

    @property
    def logstd(self):
        return self._logstd

    @property
    def std(self):
        return tf.exp(self._logstd)

# https://github.com/keras-team/keras/issues/3878
class LayerNorm(tf.keras.layers.Layer):

    def __init__(self, axis=-1, eps=1e-6, **kwargs):
        self.axis = axis
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_variable(name='gamma',
                                        shape=input_shape[1:],
                                        initializer=tf.keras.initializers.Ones(),
                                        trainable=True)
        self.beta = self.add_variable(name='beta',
                                        shape=input_shape[1:],
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        std = K.std(inputs, axis=self.axis, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

# I presume this is just how Sequential is added but at the moment Sequential requires input size to be specified at the begining
class Stack(tf.keras.Model):
    def __init__(self, layers=None):
        super().__init__()
        self._call = None
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)

    def call(self, inputs, training=None, mask=None):
        output = inputs
        for layer in self._layers:
            output = layer(output)
        return output

class Conv2DStack(Stack):

    def __init__(self, layers, batch_norm=False, activation='relu', padding='same', flatten_output=True):
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

    def __init__(self, layers, batch_norm=False, activation='relu', output_activation=None):
        super().__init__()
        if layers is None:
            layers = []
        for i, layer in enumerate(layers[:-1]):
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

    def __init__(self, layer, norm=False, normaxis=-1):
        super().__init__()
        self.norm = norm
        self.normaxis = normaxis
        self.layer = layer
        if norm:
            self.normalization = LayerNorm(normaxis)

    def call(self, inputs, *args, **kwargs):
        """Implements residual connection w/ possible normalization

            :param inputs: A Tensor

                implements -> output = LayerNorm(x + layer(x))
        """
        layer_out = self.layer(inputs, *args, **kwargs)
        if isinstance(inputs, tuple):
            inputs = inputs[0]

        residual = inputs + layer_out
        if self.norm:
            residual = self.normalization(residual)
        return residual
