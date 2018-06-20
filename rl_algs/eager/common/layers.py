import collections
import tensorflow as tf

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

    def __init__(self, layers, batch_norm=False, activation='relu'):
        super().__init__()
        if layers is None:
            layers = []
        for i, layer in enumerate(layers):
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(tf.keras.layers.Dense(*layer))
            if batch_norm:
                self.add(tf.keras.layers.BatchNormalization())
            self.add(tf.keras.layers.Activation(activation))