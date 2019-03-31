"""
Attention layers for the transformer model
"""

from typing import Optional

import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object

from rinokeras.core.v1x.common.attention import SelfAttention, MultiHeadAttention
from rinokeras.core.v1x.common.layers import LayerNorm


class TransformerSelfAttention(tf.keras.Model):

    def __init__(self,
                 n_heads: int,
                 dropout: Optional[float] = None,
                 key_size: Optional[int] = None,
                 use_residual_norm: bool = True,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer : Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer : Optional[tf.keras.regularizers.Regularizer] = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.key_size = key_size
        self.norm = LayerNorm()
        self.use_residual_norm = use_residual_norm
        self.self_attention = SelfAttention(
            'scaled_dot', n_heads, dropout,
            key_size=key_size,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, return_attention_weights=False):
        attn_inputs = self.norm(inputs) if self.use_residual_norm else inputs
        attention, attention_weights = self.self_attention(
            attn_inputs, mask=mask, return_attention_weights=True)
        
        # Residual connection on the attention block
        output = inputs + attention
        if not self.use_residual_norm:
            output = self.norm(output)

        if return_attention_weights:
            return output, attention_weights
        return output

    def get_config(self):
        config = {
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'key_size': self.key_size,
            'use_residual_norm': self.use_residual_norm,
            'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
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


class TransformerMultiAttention(tf.keras.Model):

    def __init__(self,
                 n_heads: int,
                 dropout: Optional[float] = None,
                 key_size: Optional[int] = None,
                 use_residual_norm: bool = True,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer:  Optional[tf.keras.regularizers.Regularizer] = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.key_size = key_size
        self.multi_attention = MultiHeadAttention(
            'scaled_dot', n_heads, dropout,
            key_size=key_size,
            project_value=True,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer
        )
        self.norm = LayerNorm()
        self.use_residual_norm = use_residual_norm

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, return_attention_weights=False):
        source, target = inputs
        attn_inputs = self.norm(target) if self.use_residual_norm else target
        attention, attention_weights = self.multi_attention(
            (attn_inputs, source, source), mask=mask, return_attention_weights=True)

        # Residual connection on the target
        output = target + attention
        if not self.use_residual_norm:
            output = self.norm(output)

        if return_attention_weights:
            return output, attention_weights
        return output

    def get_config(self):
        config = {
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'key_size': self.key_size,
            'use_residual_norm': self.use_residual_norm,
            'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
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
