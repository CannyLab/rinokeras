from typing import Optional
import tensorflow as tf
import numpy as np

from rl_algs.eager.common.layers import Stack, DenseStack, LayerNorm, PositionEmbedding, Highway
from rl_algs.eager.common.attention import ContextQueryAttention, SelfAttention


class QANetSelfAttention(tf.keras.Model):
    def __init__(self, n_heads: int, dropout: Optional[float]) -> None:
        """QANet Self-Attention block

        Arguments:
            n_heads {int} -- The number of heads in the attention block
            dropout {Optional[float]} -- Optional dropout constant
        """

        super().__init__()
        self.self_attention = SelfAttention('scaled_dot', n_heads, dropout)
        self.norm = LayerNorm()

    def call(self, inputs, mask):
        norm_input = self.norm(inputs)
        attention = self.self_attention(norm_input, mask=mask)
        return attention + inputs  # Just do the residual connection manually


class QANetFeedForward(tf.keras.Model):
    def __init__(self, filter_size: int, hidden_size: int, dropout: Optional[float]) -> None:
        """QANet Feed Forward block

        Arguments:
            filter_size {int} -- The size of the filter
            hidden_size {int} -- The size of the hidden layer
            dropout {Optional[float]} -- Optional dropout constant
        """

        super().__init__()
        dense_relu_dense = DenseStack(
            [filter_size, hidden_size], output_activation=None)
        if dropout is not None:
            dropout = tf.keras.layers.Dropout(dropout)
            dense_relu_dense = Stack([dense_relu_dense, dropout])
        self.feed_forward = dense_relu_dense
        self.norm = LayerNorm()

    def call(self, inputs):
        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        return dense_out + inputs


class QANetConvBlock(tf.keras.Model):
    """
    Layered depth-wise separable convolutions.

    Based on https://arxiv.org/pdf/1804.09541.pdf.
    """

    def __init__(self, filters: int, kernel_size: int, dropout: Optional[float]) -> None:
        """QANet Convolutional block

        Arguments:
            filters {int} -- The number of filters in the block
            kernel_size {int} -- The size of the kernel for the block
            dropout {Optional[float]} -- Optional dropout constant
        """

        super().__init__()
        conv_layer = tf.keras.layers.SeparableConv1D(
            filters, kernel_size, padding='same')
        if dropout is not None:
            dropout = tf.keras.layers.Dropout(dropout)
            conv_layer = Stack([conv_layer, dropout])
        self.conv_layer = conv_layer
        self.norm = LayerNorm()

    def call(self, inputs, mask):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, seqlen, d_model]
                mask: a float32 Tensor with shape [batch_size, seqlen, seqlen]
        """
        norm_input = self.norm(inputs)
        if mask is not None:
            mask = tf.cast(mask[:, 0, :], tf.float32)
            norm_input = norm_input * mask[:, :, None]
        conv_out = self.conv_layer(norm_input)

        return conv_out + inputs


class QANetInputEmbedding(tf.keras.Model):

    # TODO: Add character level embedding
    def __init__(self,
                 d_model: int,
                 word_embed_initializer: np.ndarray,
                 char_embed_initializer: np.ndarray,
                 dropout: Optional[float] = None,
                 batch_norm: bool = False) -> None:
        """QANet Imput embedding class

        Arguments:
            d_model {int} -- The model dimension
            word_embed_initializer {np.ndarray} -- The word-level embedding matrix
            char_embed_initializer {np.ndarray} -- The character-level embedding matrix

        Keyword Arguments:
            dropout {Optional[float]} -- Dropout constant in the embedding (default: {None})
            batch_norm {bool} -- Use batch normalization in the embedding (default: {False})
        """

        super().__init__()
        self.word_embedding = tf.keras.layers.Embedding(word_embed_initializer.shape[0],
                                                        word_embed_initializer.shape[1],
                                                        weights=[
                                                            word_embed_initializer],
                                                        mask_zero=True)
        self.char_embedding = tf.keras.layers.Embedding(char_embed_initializer.shape[0],
                                                        char_embed_initializer.shape[1],
                                                        weights=[
                                                            char_embed_initializer],
                                                        mask_zero=True)
        self.char_conv = tf.keras.layers.Conv1D(
            filters=char_embed_initializer.shape[1], kernel_size=5)
        self.projection_conv = tf.keras.layers.Conv1D(
            filters=d_model, kernel_size=1)

        self.highway = Stack([Highway(dropout=dropout) for _ in range(2)])

        self.position_embedding = PositionEmbedding()
        self.dropout = None if dropout is None else tf.keras.layers.Dropout(
            dropout)
        self.batch_norm = None if batch_norm is False else tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        """
        Args:
            words: a Tensor with shape (batch_size, input_length)
            chars: a Tensor with shape (batch_size, input_length, 16)

        Returns:
            embedding: an embedding of the input with shape (batch_size, input_length, d_model)
        """
        words, chars = inputs
        # word_embedding -> Tensor with shape (batch_size, input_length, 300)
        word_embedding = self.word_embedding(words)
        # char_embedding -> Tensor with shape (batch_size, input_length, 16, 200)
        char_embedding = self.char_embedding(chars)
        if self.dropout:
            word_embedding = self.dropout(word_embedding)
            char_embedding = self.dropout(char_embedding)
        # char_embedding -> Tensor with shape (batch_size * input_length, 16, 200)
        char_embedding = tf.reshape(char_embedding, (char_embedding.shape[0] * char_embedding.shape[1],
                                                     char_embedding.shape[2],
                                                     char_embedding.shape[3]))
        # char_embedding -> Tensor with shape (batch_size * input_length, 16, n_filters)
        char_embedding = self.char_conv(char_embedding)
        # char_embedding -> Tensor with shape (batch_size, input_length, n_filters)
        char_embedding = tf.reduce_max(char_embedding, -2)
        char_embedding = tf.reshape(char_embedding, (word_embedding.shape[0],
                                                     word_embedding.shape[1],
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


class QANetEncoderBlock(tf.keras.Model):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

    def __init__(self,
                 n_conv: int,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float] = None) -> None:
        """QANet Encoder Block

        Arguments:
            n_conv {int} -- Number of convolutions in the encoder layer
            n_heads {int} -- Number of heads in the self-attention
            filter_size {int} -- Filter size in the feed-forward layer
            hidden_size {int} -- Hidden layer size in the feed-forward layer/conv block

        Keyword Arguments:
            dropout {Optional[float]} -- Optional dropout constant (default: {None})
        """

        super().__init__()
        self.conv_stack = Stack(
            [QANetConvBlock(hidden_size, 7, dropout) for _ in range(n_conv)])
        self.self_attention = QANetSelfAttention(n_heads, dropout)
        self.feed_forward = QANetFeedForward(filter_size, hidden_size, dropout)

    def call(self, inputs, self_attention_mask=None, padding_mask=None):
        # if self_attention_mask is not None and padding_mask is None:
            # padding_mask = self_attention_mask
        conv_out = self.conv_stack(inputs, mask=padding_mask)
        res_attn = self.self_attention(conv_out, mask=self_attention_mask)
        output = self.feed_forward(res_attn)
        return output


class QANet(tf.keras.Model):

    def __init__(self,
                 word_embed_matrix: np.ndarray,
                 char_embed_matrix: np.ndarray,
                 n_heads: int = 8,
                 d_model: int = 128,
                 d_filter: int = 512,
                 char_limit: int = 16,
                 dropout: Optional[float] = None) -> None:
        """QANet (Based on https://arxiv.org/abs/1804.09541)

        Arguments:
            word_embed_matrix {np.ndarray} -- Word-level embedding matrix
            char_embed_matrix {np.ndarray} -- Character-level embedding matrix

        Keyword Arguments:
            n_heads {int} -- Number of self-attention heads (default: {8})
            d_model {int} -- Internal dimension of the model (default: {128})
            d_filter {int} -- internal dimension of the filters (default: {512})
            char_limit {int} -- Character limit for each word (default: {16})
            dropout {Optional[float]} -- Optional dropout constant (default: {None})
        """

        super().__init__()
        self.n_symbols_in = word_embed_matrix.shape[0]
        self.n_symbols_out = word_embed_matrix.shape[0]  # TODO: Fix this bug?
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.char_limit = char_limit
        self.dropout = dropout

        self.input_embedding = QANetInputEmbedding(self.d_model,
                                                   word_embed_matrix,
                                                   char_embed_matrix,
                                                   dropout=self.dropout,
                                                   batch_norm=False)
        self.embedding_encoder = Stack([QANetEncoderBlock(n_conv=4,
                                                          n_heads=self.n_heads,
                                                          filter_size=self.d_filter,
                                                          hidden_size=self.d_model,
                                                          dropout=self.dropout) for _ in range(1)])

        self.context_query_attention = ContextQueryAttention(
            None, self.d_model)

        self.model_encoder_projection = tf.keras.layers.Conv1D(filters=d_model,
                                                               kernel_size=1)
        self.model_encoder = Stack([QANetEncoderBlock(n_conv=2,
                                                      n_heads=self.n_heads,
                                                      filter_size=self.d_filter,
                                                      hidden_size=self.d_model,
                                                      dropout=self.dropout) for _ in range(7)])

        self.output_layer = tf.keras.layers.Dense(self.n_symbols_out)

    def call(self, inputs, padding_mask=None, shift_target_sequence_right=True, training=True):
        context, question, context_characters, question_characters, answer_index1, answer_index2, _ = inputs

        def get_mask_and_length(array):
            mask = tf.cast(array, tf.bool)
            length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
            return mask, length

        context_mask, context_length = get_mask_and_length(context)
        question_mask, question_length = get_mask_and_length(question)

        context_maxlen = tf.reduce_max(context_length)
        question_maxlen = tf.reduce_max(question_length)

        context = context[:, :context_maxlen]
        question = question[:, :question_maxlen]
        context_mask = context_mask[:, :context_maxlen]
        question_mask = question_mask[:, :question_maxlen]
        context_characters = context_characters[:, :context_maxlen]
        question_characters = question_characters[:, :question_maxlen]
        answer_index1 = answer_index1[:, :context_maxlen]
        answer_index2 = answer_index2[:, :context_maxlen]

        context_query_mask = self._convert_padding_masks_to_context_query_mask(
            question_mask, context_mask)
        context_mask = self._convert_padding_mask_to_attention_mask(
            context, context_mask)
        question_mask = self._convert_padding_mask_to_attention_mask(
            question, question_mask)

        context_embedding = self.input_embedding((context, context_characters))
        question_embedding = self.input_embedding(
            (question, question_characters))

        context_encoding = self.embedding_encoder(
            context_embedding, self_attention_mask=context_mask, padding_mask=context_mask)
        question_encoding = self.embedding_encoder(
            question_embedding, self_attention_mask=question_mask, padding_mask=question_mask)

        context_query_attention = self.context_query_attention((question_encoding, context_encoding),
                                                               mask=context_query_mask)
        context_query_projection = self.model_encoder_projection(
            context_query_attention)
        output = self.model_encoder(
            context_query_projection, self_attention_mask=context_mask, padding_mask=context_mask)

        return output

    def _convert_padding_mask_to_attention_mask(self, inputs, mask):
        assert mask.shape[0] == inputs.shape[0], 'Mask and input batch size must match'
        assert mask.ndim == 2, 'Can only convert dimension 2 masks to dimension 3 masks'

        seqlen = inputs.shape[1]
        mask = tf.tile(mask[:, None, :], (1, seqlen, 1))
        return mask

    def _convert_seqlens_to_attention_mask(self, inputs, seqlens):
        assert seqlens.shape[0] == inputs.shape[0], 'Seqlens and input batch size must match'
        assert seqlens.ndim == 1, 'Can only convert dimension 1 seqlens to dimension 3 masks'

        indices = tf.tile(tf.range(inputs.shape[1])[
                          None, :], (seqlens.shape[0], 1))
        mask = indices < seqlens[:, None]
        return mask

    def _convert_padding_masks_to_context_query_mask(self, query_mask, context_mask):
        return tf.logical_and(context_mask[:, :, None], query_mask[:, None, :])
