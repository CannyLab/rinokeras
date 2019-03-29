"""
Positional embedding layers
"""

from typing import Dict

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense  # pylint: disable=F0401

from rinokeras.core.v1x.utils import get_shape


class PositionEmbedding(Model):
    """
    Adds positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self, concat: bool = False, reproject_embedding: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.concat = concat
        self.reproject_embedding = reproject_embedding

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        if not self.concat:
            assert hidden_size % 2 == 0, 'Model vector size must be even for sinusoidal encoding'
        else:
            if hidden_size % 2 != 0:
                hidden_size += 1
        power = tf.range(0, hidden_size.value, 2,
                         dtype=tf.float32) / hidden_size.value
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

        if self.reproject_embedding:
            self.projection_layer = Dense(input_shape[-1])

    def call(self, inputs, start=1):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        inputs.shape.assert_has_rank(3)
        if not self.concat:
            assert inputs.shape[-1] == self.hidden_size, 'Input final dim must match model hidden size'
        batch_size = get_shape(inputs, 0)
        sequence_length = get_shape(inputs, 1)
        seq_pos = tf.cast(tf.range(start, sequence_length + start)
                          [None, :], tf.float32)  # 1-index positions

        index = seq_pos[:, :, None] / self.divisor

        sin_embedding = tf.sin(index)
        cos_embedding = tf.cos(index)

        position_embedding = tf.stack((sin_embedding, cos_embedding), -1)
        position_shape = (1, sequence_length, self.hidden_size)

        position_embedding = tf.reshape(position_embedding, position_shape)

        if self.concat:
            position_embedding = tf.tile(position_embedding, (batch_size, 1, 1))
            output = tf.concat((inputs, position_embedding), -1)
            # Return the reprojection to the hidden size of the layer, if we're doing
            # that, otherwise, just return the layer
            if self.reproject_embedding:
                output = self.projection_layer(output)
            return output

        return inputs + position_embedding

    def compute_output_shape(self, input_shape):
        if not self.concat or self.reproject_embedding:
            return input_shape
        else:
            if isinstance(input_shape, tf.TensorShape):
                input_shape = input_shape.as_list()
            return tf.TensorShape(input_shape[:-1] + [2 * input_shape[-1]])

    def get_config(self) -> Dict:
        return dict()


class PositionEmbedding2D(PositionEmbedding):
    """
    Adds a 2D positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self, concat: bool = False, **kwargs):
        super().__init__(concat, **kwargs)

    def build(self, input_shape):
        # self.embedding = self.add_weight('embedding', input_shape.as_list()[1:], dtype=tf.float32, trainable=True)
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()
        hidden_size = input_shape[-1]
        assert hidden_size % 4 == 0, 'Model vector size must be multiple of four for 2D sinusoidal encoding'

        power = tf.range(0, hidden_size, 4,
                         dtype=tf.float32) / hidden_size
        divisor = 1000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

        if self.reproject_embedding:
            self.projection_layer = Dense(input_shape[-1])

    def call(self, inputs, start=None):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, Width, Height, Channels]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, Width, Height, Channels]
        """
        # return inputs + self.embedding[None]
        inputs.shape.assert_has_rank(4)
        batch_size = get_shape(inputs, 0)
        width = get_shape(inputs, 1)
        height = get_shape(inputs, 2)
        channels = inputs.shape[-1]
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

        if self.concat:
            position_embedding = tf.tile(
                position_embedding, (batch_size, 1, 1, 1))
            output = tf.concat((inputs, position_embedding), -1)
            if self.reproject_embedding:
                output = self.projection_layer(output)
            return output
        else:
            return inputs + position_embedding


class PositionEmbedding3D(PositionEmbedding2D):
    """
    Adds a 3D positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self, concat: bool = False, **kwargs):
        super().__init__(concat, **kwargs)

    def build(self, input_shape):
        # self.embedding = self.add_weight('embedding', input_shape.as_list()[1:], dtype=tf.float32, trainable=True)
        hidden_size = input_shape[-1]
        assert hidden_size % 6 == 0, 'Model vector size must be multiple of six for 3D sinusoidal encoding'

        power = tf.range(0, hidden_size.value, 6,
                         dtype=tf.float32) / hidden_size.value
        divisor = 1000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

        if self.reproject_embedding:
            self.projection_layer = Dense(input_shape[-1])

    def call(self, inputs, start=None):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, Width, Height, Channels]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, Width, Height, Channels]
        """
        inputs.shape.assert_has_rank(5)
        batch_size = get_shape(inputs, 0)
        time = get_shape(inputs, 1)
        width = get_shape(inputs, 2)
        height = get_shape(inputs, 3)
        channels = inputs.shape[-1]
        assert channels == self.hidden_size, 'Input final dim must match model hidden size'

        time_pos = tf.cast(tf.range(1, time + 1)[None, :], tf.float32)
        width_pos = tf.cast(tf.range(1, width + 1)[None, :], tf.float32)
        height_pos = tf.cast(tf.range(1, height + 1)[None, :], tf.float32)

        time_embed = time_pos[:, :, None] / self.divisor
        width_embed = width_pos[:, :, None] / self.divisor
        height_embed = height_pos[:, :, None] / self.divisor

        time_embed = tf.tile(time_embed[:, :, None, None, :], (1, 1, width, height, 1))
        width_embed = tf.tile(width_embed[:, None, :, None, :], (1, time, 1, height, 1))
        height_embed = tf.tile(height_embed[:, None, None, :, :], (1, time, width, 1, 1))

        time_sin_embed = tf.sin(time_embed)
        time_cos_embed = tf.cos(time_embed)
        width_sin_embed = tf.sin(width_embed)
        width_cos_embed = tf.cos(width_embed)
        height_sin_embed = tf.sin(height_embed)
        height_cos_embed = tf.cos(height_embed)

        position_embedding = tf.stack((time_sin_embed, time_cos_embed,
                                       width_sin_embed, width_cos_embed,
                                       height_sin_embed, height_cos_embed), -1)
        position_embedding = tf.reshape(
            position_embedding, (1, time, width, height, self.hidden_size))
        if self.concat:
            position_embedding = tf.tile(position_embedding, (batch_size, 1, 1, 1, 1))
            output = tf.concat((inputs, position_embedding), -1)
            if self.reproject_embedding:
                return self.projection_layer(output)
            return output
        else:
            return inputs + position_embedding


class LearnedEmbedding(Layer):
    """
    Adds learned positional embedding to an input embedding.
    """

    def __init__(self, concat: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.concat = concat

    def build(self, input_shape):
        shape = input_shape[1:]
        shape.assert_is_fully_defined()
        shape = [1] + shape.as_list()
        self.embedding = self.add_weight('embedding', shape, dtype=tf.float32, initializer='glorot_uniform')

    def call(self, inputs):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        inputs.shape[1:].assert_is_compatible_with(self.embedding.shape[1:])
        if self.concat:
            batch_size = get_shape(inputs, 0)
            embedding = tf.tile(self.embedding, [batch_size] + [1] * (inputs.shape.ndims - 1))
            return tf.concat((inputs, embedding), -1)
        else:
            return inputs + self.embedding

    def get_config(self) -> Dict:
        return dict()
