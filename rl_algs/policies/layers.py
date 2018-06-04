import os
import numpy as np
from functools import reduce
import collections

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
import tensorflow.contrib.eager as tfe

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

    def __init__(self, layers, batch_norm=False, activation='relu', flatten_output=True):
        super().__init__()
        if layers is None:
            layers = []
        def cast_layer(inputs):
            if inputs.dtype == tf.uint8:
                return tf.cast(inputs, tf.float32) / 255
        self.add(cast_layer)
        for layer in layers:
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(tf.keras.layers.Conv2D(*layer))
            if batch_norm:
                self.add(tf.keras.layers.BatchNormalization())
            self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.Flatten())

class DenseStack(Stack):

    def __init__(self, layers, activation='relu'):
        super().__init__()
        if layers is None:
            layers = []
        for i, layer in enumerate(layers):
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(tf.keras.layers.Dense(*layer))
            self.add(tf.keras.layers.Activation(activation))

class SimpleNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.all_layers = [tf.keras.layers.Dense(10, activation='relu')]
        self.all_layers.append(tf.keras.layers.Dense(2))

    def call(self, input_data):
        output = input_data
        for layer in self.all_layers:
            output = layer(output)
        return output

class MNISTModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv = Conv2DStack([(16, 5, 2), (32, 5, 2)])
        self.classify = tf.keras.layers.Dense(10)

    def call(self, inputs):
        net = self.conv(inputs)
        logits = self.classify(net)
        output = tf.nn.softmax(logits)
        return output
