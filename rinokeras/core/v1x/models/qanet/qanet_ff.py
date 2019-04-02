"""
Feed forward/Conv layers for QANet
"""

from typing import Optional
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, SeparableConv1D

from rinokeras.core.v1x.common import DenseStack, LayerNorm

class QANetFeedForward(Model):
    """QANet Feed Forward Block

    :param filter_size: The size of the input filter
    :type filter_size: int
    :param hidden_size: The size of the hidden layer
    :type hidden_size: int
    :param dropout: Dropout Weight
    :type dropout: Optional[float]

    """
    def __init__(self,
                 filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        
        self.norm = LayerNorm()
        self.feed_forward = DenseStack(
            [filter_size, hidden_size], output_activation=None,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        self.dropout_rate = dropout
        self.dropout = Dropout(0 if dropout is None else dropout)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    def call(self, inputs):
        """Compute a feed-forward pass on the inputs

        :param inputs: Input tensor
        :type inputs: tf.Tensor
        :return: Feed-Forward Output
        :rtype: tf.Tensor
        """
        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        dense_out = self.dropout(dense_out)
        return dense_out + inputs

    def get_config(self):
        config = {
            'filter_size': self.filter_size,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout_rate,
            'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class QANetConvBlock(Model):
    """QANet Convolutional Block

    Layered depth-wise separable convolutions. Based on https://arxiv.org/pdf/1804.09541.pdf.

    :param filters: The number of filters in the convolution
    :type filters: int
    :param kernel_size: The size of the convolutional kernel
    :type kernel_size: int
    :param dropout: Dropout weight
    :type dropout: Optional[float]

    """

    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.norm = LayerNorm()
        self.conv_layer = SeparableConv1D(filters, kernel_size, padding='same',
                                          depthwise_regularizer=kernel_regularizer,
                                          pointwise_regularizer=kernel_regularizer)
        
        self.dropout_rate = dropout
        self.dropout = Dropout(0 if dropout is None else dropout)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None):
        """
        :param inputs: a float32 Tensor with shape [batch_size, seqlen, d_model]
        :type inputs: tf.Tensor
        :param mask: a float32 Tensor with shape [batch_size, seqlen, seqlen]
        :type mask: tf.Tensor
        :return: a float32 Tensor with shape  [TODO (roshan_rao@berkeley.edu)]
        :rtype: tf.Tensor
        """
        norm_input = self.norm(inputs)
        if mask is not None:
            mask = tf.cast(mask[:, 0, :], norm_input.dtype)
            norm_input = norm_input * mask[:, :, None]

        conv_out = self.conv_layer(norm_input)
        conv_out = self.dropout(conv_out)

        return conv_out + inputs

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
            'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
