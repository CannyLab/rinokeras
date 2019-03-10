import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Lambda, Reshape
from tensorflow.keras.initializers import Orthogonal
import math

from rinokeras.common.layers import Stack, Conv2DStack, PositionEmbedding2D
from rinokeras.common.layers import WeightNormDense as Dense


class NatureCNN(Model):

    def __init__(self, use_rmc: bool = False):
        super().__init__()
        self.use_rmc = use_rmc
        self.prepare_input = Stack()
        self.prepare_input.add(Lambda(lambda x: tf.to_float(x) / 255))
        # self.prepare_input.add(BatchNormalization())
        self.conv_net = Conv2DStack(
            [32, 64, 64], [8, 4, 3], [4, 2, 1],
            activation='relu',
            padding='valid',
            flatten_output=not use_rmc,
            kernel_initializer=Orthogonal(math.sqrt(2.0)))
        self.prepare_output = Stack()
        if not use_rmc:
            self.prepare_output.add(
                Dense(512, activation='relu', kernel_initializer=Orthogonal(math.sqrt(2))))
        else:
            self.prepare_output.add(Dense(64, activation='relu', kernel_initializer=Orthogonal(math.sqrt(2))))
            # self.prepare_output.add(BatchNormalization())
            self.prepare_output.add(PositionEmbedding2D(concat=True))
            self.prepare_output.add(Reshape((49, 128)))

    def call(self, inputs):
        assert inputs.dtype == tf.uint8
        return self.prepare_output(self.conv_net(self.prepare_input(inputs)))
