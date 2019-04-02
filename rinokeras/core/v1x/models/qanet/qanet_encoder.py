
from typing import Optional
import tensorflow as tf
from tensorflow.keras import Model

from rinokeras.core.v1x.common import LayerDropoutStack, LayerDropout
from rinokeras.core.v1x.models.qanet import QANetConvBlock, QANetFeedForward, QANetSelfAttention


class QANetEncoderBlock(Model):
    """QANet Encoder Block
    """

    def __init__(self,
                 n_conv: int,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 kernel_size: int = 7,
                 dropout: Optional[float] = None,
                 layer_dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.n_conv = n_conv
        self.n_heads = n_heads
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.layer_dropout = layer_dropout

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(
            activity_regularizer)

        self.conv_stack = LayerDropoutStack([QANetConvBlock(hidden_size, kernel_size, dropout, kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                activity_regularizer=activity_regularizer) for _ in range(n_conv)], layer_dropout=0 if layer_dropout is None else layer_dropout)
        self.self_attention = QANetSelfAttention(n_heads, dropout,
                                                 kernel_regularizer=kernel_regularizer,
                                                 bias_regularizer=bias_regularizer,
                                                 activity_regularizer=activity_regularizer)
        self.layer_drop_2 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)
        self.feed_forward = QANetFeedForward(filter_size, hidden_size, dropout,
                                             kernel_regularizer=kernel_regularizer,
                                             bias_regularizer=bias_regularizer,
                                             activity_regularizer=activity_regularizer)
        self.layer_drop_3 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)

    def call(self, inputs, mask=None):
        """Computes the encoding on the context

        :param inputs: The inputs to compute over
        :type inputs: tf.Tensor
        :param self_attention_mask: Self Attention Mask, defaults to None
        :param self_attention_mask: tf.Tensor, optional
        :param padding_mask: Padding Mask, defaults to None
        :param padding_mask: tf.Tensor, optional
        :return: The convolutional stack + the self-attention
        :rtype: tf.Tensor
        """

        if mask is not None:
            self_attention_mask, padding_mask = mask
        else:
            self_attention_mask, padding_mask = (None, None)

        conv_out = self.conv_stack(inputs, mask=padding_mask) 
        res_attn = self.layer_drop_2( self.self_attention(conv_out), conv_out, mask=self_attention_mask)
        output = self.layer_drop_3(self.feed_forward(res_attn),res_attn)
        # def subcall(inputs):
        # conv_out = self.conv_stack(inputs, mask=padding_mask)
        # res_attn = self.self_attention(conv_out, mask=self_attention_mask)
        # output = self.feed_forward(res_attn)
        # return output
        # output = self.layer_drop(subcall, inputs)
        return output

    def get_config(self):
        config = {
            'n_conv': self.n_conv,
            'n_heads': self.n_heads,
            'filter_size': self.filter_size,
            'hidden_size': self.hidden_size,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'layer_dropout': self.layer_dropout,
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
