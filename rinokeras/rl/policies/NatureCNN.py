import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import Orthogonal
import math

from rinokeras.common.layers import Stack, Conv2DStack, PositionEmbedding2D
from rinokeras.common.layers import WeightNormDense as Dense


class NatureCNN(Model):

    def __init__(self, use_rmc: bool = False):
        super().__init__()
        self.use_rmc = use_rmc
        self.forward = Stack()
        self.forward.add(BatchNormalization())
        self.forward.add(Conv2DStack(
            [32, 64, 64], [8, 4, 3], [4, 2, 1],
            activation='relu',
            padding='valid',
            flatten_output=not use_rmc,
            kernel_initializer=Orthogonal(math.sqrt(2.0))))
        if not use_rmc:
            self.forward.add(
                Dense(512, activation='relu', kernel_initializer=Orthogonal(math.sqrt(2))))
        else:
            self.forward.add(Dense(64, activation='relu', kernel_initializer=Orthogonal(math.sqrt(2))))
            self.forward.add(PositionEmbedding2D(concat=True))

    def call(self, inputs):
        assert inputs.dtype == tf.uint8
        norm = tf.cast(inputs, tf.float32) / 255.
        output = self.forward(norm)
        if self.use_rmc:
            batch_size = tf.shape(output)[0]
            height = output.shape[1]
            width = output.shape[2]
            filters = output.shape[3]
            output = tf.reshape(output, (batch_size, height * width, filters))
        return output
