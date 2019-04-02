
import tensorflow as tf
from typing import Tuple
from rinokeras.core.v1x.common.layers import LayerNorm, Stack
from rinokeras.core.v1x.common.layers.conv import GroupedConvolution

class ResidualBlock(tf.keras.Model):
    def __init__(self, cardinality: int = 1, n_filters_in: int = 64, n_filters_out: int = 64, kernel_size: Tuple[int, int] = (3, 3), stride: Tuple[int, int] = (1, 1),
                 use_layernorm: bool = True, activation_fn='relu', project_shortcut=False, alpha=0.3) -> None:
        super(ResidualBlock, self).__init__()

        self.stride = stride
        self.use_layernorm = use_layernorm
        self.project_shortcut = project_shortcut

        # Build the bottleneck
        self.conv_1 = tf.keras.layers.Conv2D(filters=n_filters_in, kernel_size=(1,1), strides=(1,1), padding='same')
        if self.use_layernorm:
            self.layer_norm_1 = LayerNorm()

        valid_activations = ['relu']
        if activation_fn not in valid_activations:
            raise ValueError('Unrecognized activation function: {}, must be one of {}'.format(activation_fn, valid_activations))

        if activation_fn == 'relu':
            self.internal_activation = tf.keras.layers.LeakyReLU(alpha)

        # Build the second convolutional block
        self.conv_2 = GroupedConvolution(cardinality=cardinality, n_filters=n_filters_in, kernel_size=kernel_size, stride=self.stride)
        if self.use_layernorm:
            self.layer_norm_2 = LayerNorm()

        # Build the output bottleneck
        self.conv_3 = tf.keras.layers.Conv2D(filters=n_filters_out, kernel_size=(1,1), strides=(1,1), padding='same')
        if self.use_layernorm:
            self.layer_norm_3 = LayerNorm()

        # Shortcut projection
        if project_shortcut or self.stride != (1,1):
            self.projection_conv = tf.keras.layers.Conv2D(filters=n_filters_out, kernel_size=(1,1), strides=self.stride, padding='same')
            if self.use_layernorm:
                self.layer_norm_projection = LayerNorm()

    def call(self, inputs, *args, **kwargs):

        shortcut = inputs

        # Bottleneck
        residual = self.conv_1(inputs)
        if self.use_layernorm:
            residual = self.layer_norm_1(residual)
        residual = self.internal_activation(residual)

        # Grouped Convolution
        residual = self.conv_2(residual)
        if self.use_layernorm:
            residual = self.layer_norm_2(residual)
        residual = self.internal_activation(residual)

        # Bottleneck out
        residual = self.conv_3(residual)
        if self.use_layernorm:
            residual = self.layer_norm_3(residual)

        # Project shortcut if necessary
        if self.project_shortcut or self.stride != (1, 1):
            shortcut = self.projection_conv(shortcut)
            if self.use_layernorm:
                shortcut = self.layer_norm_projection(shortcut)

        # Output
        return self.internal_activation(shortcut + residual)


# Designed for 224 x 224 x 3 inputs
class ResNeXt50(Stack):
    def __init__(self, use_layer_norm: bool = True) -> None:
        super(ResNeXt50, self).__init__()

        self.cardinality = 32

        # Conv 1
        self.add(tf.keras.layers.Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same'))
        if use_layer_norm:
            self.add(LayerNorm())
        self.add(tf.keras.layers.LeakyReLU())

        # Conv2
        self.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same'))
        for i in range(3):
            project_shortcut = (i == 0)
            self.add(ResidualBlock(self.cardinality, n_filters_in=128, n_filters_out=256, project_shortcut=project_shortcut))

        # Conv3
        for i in range(4):
            strides = (2,2) if i == 0 else (1,1)
            self.add(ResidualBlock(self.cardinality, n_filters_in=256, n_filters_out=512, stride=strides))

        # Conv4
        for i in range(6):
            strides = (2,2) if i == 0 else (1,1)
            self.add(ResidualBlock(self.cardinality, n_filters_in=512, n_filters_out=1024, stride=strides))

        # Conv5
        for i in range(3):
            strides = (2,2) if i == 0 else (1,1)
            self.add(ResidualBlock(self.cardinality, n_filters_in=1024, n_filters_out=2048, stride=strides))

        self.add(tf.keras.layers.GlobalAveragePooling2D())
        # self.add(tf.keras.layers.Dense(1))