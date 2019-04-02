"""
QANet embedding layer
"""


from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv1D, Embedding

from rinokeras.core.v1x.common import (Dropout, Highway, PositionEmbedding,
                                       Stack)
from rinokeras.core.v1x.common import WeightNormDense as Dense


class QANetInputEmbedding(Model):
    """QANet Input Embedding Class

    Perform an embedding of the input vector based on the words and characters.

    :param d_model: The dimension of the model (Output dim)
    :type d_model: int
    :param word_embed_initializer: Initialization matrix of size [word vocab size x d_model]
    :type word_embed_initializer: np.ndarray
    :param char_embed_initializer: Initialization matrix of size [char vocab size x d_model]
    :type char_embed_initializer: np.ndarray
    :param dropout: The dropout weight, defaults to None
    :param dropout: Optional[float], optional
    :param batch_norm: Use batch normalization, defaults to False
    :param batch_norm: bool, optional

    """

    def __init__(self,
                 d_model: int,
                 word_embed_initializer: np.ndarray,
                 char_embed_initializer: np.ndarray,
                 dropout: Optional[float] = None,
                 batch_norm: bool = False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.word_embed_initializer = word_embed_initializer
        self.char_embed_initializer = char_embed_initializer
        self.word_embedding = Embedding(word_embed_initializer.shape[0],
                                        word_embed_initializer.shape[1],
                                        weights=[word_embed_initializer],
                                        mask_zero=True)
        self.char_embedding = Embedding(char_embed_initializer.shape[0],
                                        char_embed_initializer.shape[1],
                                        weights=[char_embed_initializer],
                                        mask_zero=True)
        self.char_conv = Conv1D(filters=char_embed_initializer.shape[1], kernel_size=5,
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,
                                activity_regularizer=activity_regularizer)
        self.projection_conv = Conv1D(filters=d_model, kernel_size=1,
                                      kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activity_regularizer=activity_regularizer)

        self.highway = Stack(
            [Highway(Dense(d_model), dropout=dropout) for _ in range(2)])

        self.dropout_rate = dropout
        self.dropout = Dropout(0 if dropout is None else dropout)
        self.use_batch_norm = batch_norm
        self.batch_norm = None if batch_norm is False else BatchNormalization()
        self.position_embedding = PositionEmbedding()

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(
            activity_regularizer)

    def call(self, inputs):
        """Calls the input embedding on the new inputs

        Computes a set of table lookups with the passed in word and character embeddings.

        :param inputs: Tuple of (Words, Characters)
        :type inputs: Tuple[tf.Tensor, tf.Tensor]
        :return: Embedded Inputs
        :rtype: tf.Tensor
        """

        words, chars = inputs
        batch_size = tf.shape(words)[0]
        input_length = tf.shape(words)[1]
        # word_embedding -> Tensor with shape (batch_size, input_length, 300)
        word_embedding = self.word_embedding(words)
        # char_embedding -> Tensor with shape (batch_size, input_length, 16, 200)
        char_embedding = self.char_embedding(chars)
        word_embedding = self.dropout(word_embedding)
        char_embedding = self.dropout(char_embedding)
        # char_embedding -> Tensor with shape (batch_size * input_length, 16, 200)
        char_embedding = tf.reshape(char_embedding, (batch_size * input_length,
                                                     # These .shapes stay b/c they're constant
                                                     char_embedding.shape[2],
                                                     char_embedding.shape[3]))
        # char_embedding -> Tensor with shape (batch_size * input_length, 16, n_filters)
        char_embedding = self.char_conv(char_embedding)
        # char_embedding -> Tensor with shape (batch_size, input_length, n_filters)
        char_embedding = tf.reduce_max(char_embedding, -2)
        char_embedding = tf.reshape(char_embedding, (batch_size,
                                                     input_length,
                                                     char_embedding.shape[-1]))
        # embedding -> Tensor with shape (batch_size, input_length, 300 + n_filters)
        embedding = tf.concat((word_embedding, char_embedding), -1)
        # embedding -> Tensor with shape (batch_size, input_length, d_model)
        embedding = self.projection_conv(embedding)
        embedding = self.highway(embedding)

        if self.batch_norm:
            embedding = self.batch_norm(embedding)
        embedding = self.position_embedding(embedding)
        return embedding

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'word_embed_initializer': self.word_embed_initializer.tolist(),
            'char_embed_initializer': self.char_embed_initializer.tolist(),
            'dropout': self.dropout_rate,
            'batch_norm': self.use_batch_norm,
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
        wei = np.array(config.pop('word_embed_initializer'))
        cei = np.array(config.pop('char_embed_initializer'))
        return cls(word_embed_initializer=wei, char_embed_initializer=cei, **config)
