"""
QANet Attention Blocks
"""
import tensorflow as tf
from typing import Optional
from tensorflow.keras import Model
from rinokeras.core.v1x.common import SelfAttention, LayerNorm


class QANetSelfAttention(Model):
    """QANet Self Attention Block

    :param n_heads: The number of heads in the self attention block
    :type n_heads: int
    :param dropout: Dropout weight
    :type dropout: Optional[float]

    """

    def __init__(self,
                 n_heads: int,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        self.dropout = dropout
        self.self_attention = SelfAttention('scaled_dot', n_heads, dropout,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer)
        self.norm = LayerNorm()

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None):
        """Calls the Self-Attention module on the provided inputs

            :param inputs: The inputs to the self-attention module
            :type inputs: tf.Tensor
            :param mask: The self-attention mask
            :type mask: tf.Tensor
            :return: The self-attended inputs
            :rtype: tf.Tensor
        """
        norm_input = self.norm(inputs)
        attention = self.self_attention(norm_input, mask=mask)
        return attention + inputs  # Just do the residual connection manually

    def get_config(self):
        config = {
            'n_heads': self.n_heads,
            'dropout': self.dropout,
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
