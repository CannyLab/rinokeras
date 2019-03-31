import tensorflow as tf
from typing import Optional
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Conv1D

from rinokeras.core.v1x.common.layers import Stack, LayerNorm
from rinokeras.core.v1x.common.layers import WeightNormDense


class TransformerFeedForward(Model):

    def __init__(self, filter_size: int,
                 hidden_size: int,
                 kernel_size: int = 7,
                 dropout: Optional[float] = None,
                 use_conv: bool = False,
                 use_weight_norm: bool = True,
                 use_residual_norm: bool = True,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer:  Optional[tf.keras.regularizers.Regularizer] = None) -> None:
        super().__init__()
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_conv = use_conv
        self.use_residual_norm = use_residual_norm
        self.norm = LayerNorm()
        layer_args = {
            'kernel_initializer': kernel_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer}

        self.use_weight_norm = use_weight_norm
        if self.use_weight_norm:
            layer_type = WeightNormDense if not use_conv else Conv1D
        else:
            layer_type = Dense if not use_conv else Conv1D

        if use_conv:
            conv_args = {
                'kernel_size': kernel_size,
                'padding': 'same',
                'strides': 1}
            layer_args.update(conv_args)
        self.feed_forward = Stack()
        self.feed_forward.add(
            layer_type(filter_size, activation='relu', **layer_args))
        self.feed_forward.add(
            layer_type(hidden_size, activation='linear', **layer_args))
        self.feed_forward.add(Dropout(0 if dropout is None else dropout))

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    def call(self, inputs, padding_mask=None):
        if padding_mask is not None:
            inputs = inputs * tf.cast(padding_mask[..., None], inputs.dtype)

        ff_inputs = self.norm(inputs) if self.use_residual_norm else inputs
        dense_out = self.feed_forward(ff_inputs)

        output = inputs + dense_out
        if not self.use_residual_norm:
            output = self.norm(output)
        return output

    def get_config(self):
        config = {
            'filter_size': self.filter_size,
            'hidden_size': self.hidden_size,
            'kernel_size': self.kernel_size,
            'use_conv': self.use_conv,
            'use_weight_norm': self.use_weight_norm,
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
