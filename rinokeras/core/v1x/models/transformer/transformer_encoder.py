
import tensorflow as tf
import rinokeras
from typing import Optional, Union

from tensorflow.keras import Model
from tensorflow.python.keras import layers as layer_module

from rinokeras.core.v1x.common.layers import LayerDropout, Stack, Layer
from rinokeras.core.v1x.utils import get_shape

from .transformer_attention import TransformerSelfAttention
from .transformer_ff import TransformerFeedForward


class TransformerEncoderBlock(Model):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

    def __init__(self,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 conv_kernel_size: int = 7,
                 dropout: Optional[float] = None,
                 layer_dropout: Optional[float] = None,
                 use_conv: bool = False,
                 use_weight_norm=True,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer:  Optional[tf.keras.regularizers.Regularizer] = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.use_conv = use_conv
        self.use_weight_norm = use_weight_norm

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.self_attention = TransformerSelfAttention(
            n_heads, dropout,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        self.layer_drop_1 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)
        self.feed_forward = TransformerFeedForward(filter_size, hidden_size, 
                                                   dropout=dropout,
                                                   kernel_initializer=kernel_initializer,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                   use_conv=use_conv,
                                                   kernel_size=conv_kernel_size,
                                                   use_weight_norm=use_weight_norm)
        self.layer_drop_2 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)

    def call(self, inputs, mask=None, return_attention_weights=False):

        # Unpack the masking
        if mask is not None:
            if isinstance(mask, tuple):
                self_attention_mask, conv_mask = mask
            else:
                self_attention_mask = mask
                conv_mask = None
        else:
            self_attention_mask = None
            conv_mask = None 

        # Perform a multi-headed self-attention across the inputs.
        res_attn, attention_weights = self.self_attention(
            inputs, mask=self_attention_mask, return_attention_weights=True)
        res_attn = self.layer_drop_1(res_attn, inputs)

        output = self.feed_forward(res_attn, padding_mask=conv_mask)
        output = self.layer_drop_2(output, res_attn)

        if return_attention_weights:
            return output, attention_weights
        return output

    def get_config(self):
        config = {
            'n_heads': self.n_heads,
            'filter_size': self.filter_size,
            'hidden_size': self.hidden_size,
            'conv_kernel_size': self.conv_kernel_size,
            'dropout': self.dropout,
            'layer_dropout': self.layer_dropout,
            'use_conv': self.use_conv,
            'use_weight_norm': self.use_weight_norm,
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



class TransformerEncoder(Model):
    """
    Stack of TransformerEncoderBlocks. Performs repeated self-attention.
    """

    def __init__(self,
                 embedding_layer: Optional[Union[Model, Layer]],
                 n_layers: int,
                 n_heads: int,
                 d_model: int,
                 d_filter: int,
                 conv_kernel_size: int = 7,
                 dropout: Optional[float] = None,
                 layer_dropout: Optional[float] = None,
                 use_conv: bool = False,
                 use_weight_norm=True,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer:  Optional[tf.keras.regularizers.Regularizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        # Save the local variables for configuration
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.use_conv = use_conv
        self.use_weight_norm = use_weight_norm
        self.extra_kwargs = kwargs

        # Get the initializers for serialization
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.embedding_layer = embedding_layer
        # The encoding stack is a stack of transformer encoder blocks
        self.encoding_stack = Stack([TransformerEncoderBlock(
            n_heads=n_heads,
            filter_size=d_filter,
            hidden_size=d_model,
            dropout=dropout,
            layer_dropout=layer_dropout,
            use_conv=use_conv,
            conv_kernel_size=conv_kernel_size,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            use_weight_norm=use_weight_norm) for _ in range(n_layers)], name='encoder_stack')

    def call(self, inputs, mask=None):
        """
            Args:
                inputs: Either a float32 or in32 Tensor with shape [batch_size, sequence_length, ndim]
                encoder_mask: a boolean Tensor with shape [batch_size, sequence_length, sequence_length]
            Returns:
                output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        # Unpack the mask
        if mask is not None:
            if isinstance(mask, tuple):
                encoder_mask, conv_mask = mask
            else:
                encoder_mask = mask
                conv_mask = None
        else:
            encoder_mask = None
            conv_mask = None

        # Do the input embedding
        if self.embedding_layer is not None:
            inputs = self.embedding_layer(inputs)
            if conv_mask is not None:
                inputs = inputs * tf.cast(conv_mask[:, :, None], tf.float32)
        
        # Check the outputs
        inputs.shape.assert_has_rank(3)
        batch_size, seqlen, _ = get_shape(inputs, range(3))
        # We need to make sure that the input shapes are correct for the mask
        assertions = []     

        # Masking assertions
        if encoder_mask is not None:
            # Check the dimension of the mask
            encoder_mask.shape.assert_has_rank(3)
            enc_batch, enc_seq1, enc_seq2 = get_shape(encoder_mask, range(3))
            enc_batch_assert = tf.assert_equal(
                batch_size, enc_batch,
                message='Batch size mismatch between inputs and encoder mask')
            enc_seq1_assert = tf.assert_equal(
                seqlen, enc_seq1,
                message='Seqlen mismatch between inputs and encoder mask')
            enc_seq2_assert = tf.assert_equal(
                seqlen, enc_seq2,
                message='Seqlen mismatch between inputs and encoder mask')
            assertions += [enc_batch_assert, enc_seq1_assert, enc_seq2_assert]

        if conv_mask is not None:
            conv_mask.shape.assert_has_rank(2)
            conv_batch, conv_seq = get_shape(conv_mask, range(2))
            conv_batch_assert = tf.assert_equal(
                batch_size, conv_batch,
                message='Batch size mismatch between inputs and conv mask')
            conv_seq_assert = tf.assert_equal(
                seqlen, conv_seq,
                message='Seqlen mismatch between inputs and conv mask')
            assertions += [conv_batch_assert, conv_seq_assert]

        with tf.control_dependencies(assertions):
            output = self.encoding_stack(
                inputs, mask=(encoder_mask, conv_mask))

        return output

    def get_config(self):
        config = {
            'embedding_layer_config': {
                'class_name': self.embedding_layer.__class__.__name__,
                'config': self.embedding_layer.get_config(),
            },
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'd_filter': self.d_filter,
            'conv_kernel_size': self.conv_kernel_size,
            'dropout': self.dropout,
            'layer_dropout': self.layer_dropout,
            'use_conv': self.use_conv,
            'use_weight_norm': self.use_weight_norm,
            'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        }
        return rinokeras.compat.utils.merge_dicts(config, self.extra_kwargs)

    @classmethod
    def from_config(cls, config):
        embedding_layer = layer_module.deserialize(config.pop('embedding_layer_config'), custom_objects=globals())
        return cls(embedding_layer=embedding_layer, **config)
