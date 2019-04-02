"""
QANet Model
"""

import numpy as np
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Conv1D

import rinokeras.core.v1x as rk
from rinokeras.core.v1x.common import WeightNormDense as Dense
from rinokeras.core.v1x.common import Stack, ContextQueryAttention

from .qanet_embedding import QANetInputEmbedding
from .qanet_encoder import QANetEncoderBlock


class QANet(Model):
    """QANet Model

    Based on https://arxiv.org/abs/1804.09541

    :param word_embed_matrix: Word-Level Embedding Matrix
    :type word_embed_matrix: np.ndarray
    :param char_embed_matrix: Character-Level Embedding Matrix
    :type char_embed_matrix: np.ndarray
    :param n_heads: Number of heads to use for self-attention in the model, defaults to 8
    :param n_heads: int, optional
    :param d_model: Model internal dimenstion, defaults to 128
    :param d_model: int, optional
    :param d_filter: The internal filter dimension, defaults to 512
    :param d_filter: int, optional
    :param char_limit: The limit of characters in each word, defaults to 16
    :param char_limit: int, optional
    :param dropout: Dropout weight, defaults to None
    :param dropout: Optional[float], optional

    """

    def __init__(self,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_filter: int = 512,
                 char_limit: int = 16,
                 dropout: Optional[float] = None,
                 n_symbols: Optional[int] = None,
                 n_symbols_out: Optional[int] = None,
                 n_chars: Optional[int] = None,
                 word_embed_matrix: np.ndarray = None,
                 char_embed_matrix: np.ndarray = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_filter = d_filter
        self.char_limit = char_limit

        if n_symbols is None and word_embed_matrix is None:
            raise ValueError(
                'QANet cannot infer the number of symbols if word_embed_matrix is None and n_symbols is None')
        if n_symbols is None:
            self.n_symbols = word_embed_matrix.shape[0]
        else:
            self.n_symbols = n_symbols

        if n_chars is None and char_embed_matrix is None:
            raise ValueError(
                'QANet cannot infer the number of characters if char_embed_matrix is None and n_chars is None')

        self.n_chars = n_chars
        self.n_symbols_in = self.n_symbols
        if n_symbols_out is None:
            self.n_symbols_out = self.d_model
        else:
            self.n_symbols_out = n_symbols_out

        self.word_embed_matrix = word_embed_matrix
        self.char_embed_matrix = char_embed_matrix
        if self.word_embed_matrix is None:
            self.word_embed_matrix = np.random.sample((n_symbols, d_model))
        if self.char_embed_matrix is None:
            self.char_embed_matrix = np.random.sample((n_chars, d_model))

        self.input_embedding = QANetInputEmbedding(self.d_model,
                                                   self.word_embed_matrix,
                                                   self.char_embed_matrix,
                                                   dropout=dropout,
                                                   batch_norm=False,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer)
        self.embedding_encoder = Stack([QANetEncoderBlock(n_conv=4,
                                                          n_heads=self.n_heads,
                                                          filter_size=self.d_filter,
                                                          hidden_size=self.d_model,
                                                          dropout=dropout,
                                                          kernel_regularizer=kernel_regularizer,
                                                          bias_regularizer=bias_regularizer,
                                                          activity_regularizer=activity_regularizer) for _ in range(1)])

        self.context_query_attention = ContextQueryAttention(
            regularizer=kernel_regularizer)

        self.model_encoder_projection = Conv1D(filters=d_model,
                                               kernel_size=1,
                                               kernel_regularizer=kernel_regularizer,
                                               bias_regularizer=bias_regularizer,
                                               activity_regularizer=activity_regularizer)
        self.dropout_rate = dropout
        self.dropout = Dropout(0 if dropout is None else dropout)
        self.model_encoder = Stack([QANetEncoderBlock(n_conv=2,
                                                      n_heads=self.n_heads,
                                                      filter_size=self.d_filter,
                                                      hidden_size=self.d_model,
                                                      dropout=dropout,
                                                      kernel_regularizer=kernel_regularizer,
                                                      bias_regularizer=bias_regularizer,
                                                      activity_regularizer=activity_regularizer) for _ in range(7)])

        self.output_layer = Dense(self.n_symbols_out)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(
            activity_regularizer)

    def call(self, inputs, mask=None):
        """Calls the model on new  inputs.

        :param inputs: Tuple of (Context, Question, Context Characters, Question Characters)
        :type inputs: Tuple[tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor]
        :param padding_mask: The padding mask for the question and answer, defaults to None
        :param padding_mask: tf.Tensor, optional
        :param shift_target_sequence_right: Shift the target sequence to the right, defaults to True
        :param shift_target_sequence_right: bool, optional
        :return: Output logits of the model
        :rtype: tf.Tensor
        """

        context, question, context_characters, question_characters = inputs

        def get_mask_and_length(array):
            mask = tf.cast(array, tf.bool)
            length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
            return mask, length

        context_mask, _ = get_mask_and_length(context)
        question_mask, _ = get_mask_and_length(question)

        context_query_mask = self._convert_padding_masks_to_context_query_mask(
            question_mask, context_mask)
        context_mask = rk.utils.convert_sequence_mask_to_attention_mask(
            context, context_mask)
        question_mask = rk.utils.convert_sequence_mask_to_attention_mask(
            question, question_mask)
        context_embedding = self.input_embedding((context, context_characters))
        question_embedding = self.input_embedding(
            (question, question_characters))

        context_encoding = self.embedding_encoder(
            context_embedding, mask=(context_mask, context_mask))
        question_encoding = self.embedding_encoder(
            question_embedding, mask=(question_mask, question_mask))
        context_query_attention = self.context_query_attention(
            (context_encoding, question_encoding), mask=context_query_mask)
        context_query_projection = self.model_encoder_projection(
            context_query_attention)
        context_query_attention = self.dropout(context_query_attention)
        output = self.model_encoder(
            context_query_projection, mask=(context_mask, context_mask))

        return output

    def _convert_padding_masks_to_context_query_mask(self, query_mask, context_mask):
        return tf.logical_and(context_mask[:, :, None], query_mask[:, None, :])

    def get_config(self):
        config = {
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'char_limit': self.char_limit,
            'word_embed_matrix': self.word_embed_matrix.tolist(),
            'char_embed_matrix': self.char_embed_matrix.tolist(),
            'dropout': self.dropout_rate,
            'n_symbols': self.n_symbols,
            'n_symbols_out': self.n_symbols_out,
            'n_chars': self.n_chars,
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
        wei = np.array(config.pop('word_embed_matrix'))
        cei = np.array(config.pop('char_embed_matrix'))
        return cls(word_embed_matrix=wei, char_embed_matrix=cei, **config)
