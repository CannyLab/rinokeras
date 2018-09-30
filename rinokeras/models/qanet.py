from typing import Optional
import tensorflow as tf
import numpy as np

from rinokeras.common.layers import Stack, DenseStack, LayerNorm, PositionEmbedding, Highway
from rinokeras.common.attention import ContextQueryAttention, SelfAttention


class QANetSelfAttention(tf.keras.Model):
    """QANet Self Attention Block

    :param n_heads: The number of heads in the self attention block
    :type n_heads: int
    :param dropout: Dropout weight
    :type dropout: Optional[float]

    """

    def __init__(self, n_heads: int, dropout: Optional[float],
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        self.self_attention = SelfAttention('scaled_dot', n_heads, dropout,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer)
        self.norm = LayerNorm()

    def call(self, inputs, mask):
        """Calls the Self-Attention module on the provided inputs

        [description]

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


class QANetFeedForward(tf.keras.Model):
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
                 dropout: Optional[float],
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        dense_relu_dense = DenseStack(
            [filter_size, hidden_size], output_activation=None,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        if dropout is not None:
            dropout = tf.keras.layers.Dropout(dropout)
            dense_relu_dense = Stack([dense_relu_dense, dropout])
        self.feed_forward = dense_relu_dense
        self.norm = LayerNorm()

    def call(self, inputs):
        """Compute a feed-forward pass on the inputs

        :param inputs: Input tensor
        :type inputs: tf.Tensor
        :return: Feed-Forward Output
        :rtype: tf.Tensor
        """

        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        return dense_out + inputs


class QANetConvBlock(tf.keras.Model):
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
                 dropout: Optional[float],
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        conv_layer = tf.keras.layers.SeparableConv1D(filters, kernel_size, padding='same',
                                                     kernel_regularizer=kernel_regularizer,
                                                     bias_regularizer=bias_regularizer,
                                                     activity_regularizer=activity_regularizer)
        if dropout is not None:
            dropout = tf.keras.layers.Dropout(dropout)
            conv_layer = Stack([conv_layer, dropout])
        self.conv_layer = conv_layer
        self.norm = LayerNorm()

    def call(self, inputs, mask):
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
            mask = tf.cast(mask[:, 0, :], tf.float32)
            norm_input = norm_input * mask[:, :, None]
        conv_out = self.conv_layer(norm_input)

        return conv_out + inputs


class QANetInputEmbedding(tf.keras.Model):
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
                 activity_regularizer=None) -> None:
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
        self.char_conv = tf.keras.layers.Conv1D(filters=char_embed_initializer.shape[1], kernel_size=5,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                activity_regularizer=activity_regularizer)
        self.projection_conv = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1,
                                                      kernel_regularizer=kernel_regularizer,
                                                      bias_regularizer=bias_regularizer,
                                                      activity_regularizer=activity_regularizer)

        self.highway = Stack([Highway(dropout=dropout) for _ in range(2)])

        self.position_embedding = PositionEmbedding()
        self.dropout = None if dropout is None else tf.keras.layers.Dropout(
            dropout)
        self.batch_norm = None if batch_norm is False else tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        """Calls the input embedding on the new inputs

        Computes a set of table lookups with the passed in word and character embeddings. This also
        computes the positional embedding of the inputs.

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
        if self.dropout:
            word_embedding = self.dropout(word_embedding)
            char_embedding = self.dropout(char_embedding)
        # char_embedding -> Tensor with shape (batch_size * input_length, 16, 200)
        char_embedding = tf.reshape(char_embedding, (batch_size * input_length,
                                                     char_embedding.shape[2],  # These .shapes stay b/c they're constant
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


class QANetEncoderBlock(tf.keras.Model):
    """QANet Encoder Block

    An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf)

    :param n_conv: Number of convolutions in the encoding layer
    :type n_conv: int
    :param n_heads: Number of heads to use for self attention
    :type n_heads: int
    :param filter_size: The filter size in the feed-forward layer
    :type filter_size: int
    :param hidden_size: The number of neurons in the hidden layer/conv block
    :type hidden_size: int
    :param dropout: Dropout weight, defaults to None
    :param dropout: Optional[float], optional

    """

    def __init__(self,
                 n_conv: int,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        self.conv_stack = Stack([QANetConvBlock(hidden_size, 7, dropout,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                activity_regularizer=activity_regularizer)
                                 for _ in range(n_conv)])
        self.self_attention = QANetSelfAttention(n_heads, dropout,
                                                 kernel_regularizer=kernel_regularizer,
                                                 bias_regularizer=bias_regularizer,
                                                 activity_regularizer=activity_regularizer)
        self.feed_forward = QANetFeedForward(filter_size, hidden_size, dropout,
                                             kernel_regularizer=kernel_regularizer,
                                             bias_regularizer=bias_regularizer,
                                             activity_regularizer=activity_regularizer)

    def call(self, inputs, self_attention_mask=None, padding_mask=None):
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

        conv_out = self.conv_stack(inputs, mask=padding_mask)
        res_attn = self.self_attention(conv_out, mask=self_attention_mask)
        output = self.feed_forward(res_attn)
        return output


class QANet(tf.keras.Model):
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
                 word_embed_matrix: np.ndarray,
                 char_embed_matrix: np.ndarray,
                 n_heads: int = 8,
                 d_model: int = 128,
                 d_filter: int = 512,
                 char_limit: int = 16,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
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
                                                   batch_norm=False,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer)
        self.embedding_encoder = Stack([QANetEncoderBlock(n_conv=4,
                                                          n_heads=self.n_heads,
                                                          filter_size=self.d_filter,
                                                          hidden_size=self.d_model,
                                                          dropout=self.dropout,
                                                          kernel_regularizer=kernel_regularizer,
                                                          bias_regularizer=bias_regularizer,
                                                          activity_regularizer=activity_regularizer) for _ in range(1)])

        self.context_query_attention = ContextQueryAttention()

        self.model_encoder_projection = tf.keras.layers.Conv1D(filters=d_model,
                                                               kernel_size=1,
                                                               kernel_regularizer=kernel_regularizer,
                                                               bias_regularizer=bias_regularizer,
                                                               activity_regularizer=activity_regularizer)
        self.model_encoder = Stack([QANetEncoderBlock(n_conv=2,
                                                      n_heads=self.n_heads,
                                                      filter_size=self.d_filter,
                                                      hidden_size=self.d_model,
                                                      dropout=self.dropout,
                                                      kernel_regularizer=kernel_regularizer,
                                                      bias_regularizer=bias_regularizer,
                                                      activity_regularizer=activity_regularizer) for _ in range(7)])

        self.output_layer = tf.keras.layers.Dense(self.n_symbols_out)

    def call(self, inputs, padding_mask=None, shift_target_sequence_right=True, training=True):
        """Calls the model on new inputs.

        :param inputs: Tuple of (Context, Question, Context Characters, Question Characters,
                                 Answer Index 1, Answer Index 2, None)
        :type inputs: Tuple[tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor]
        :param padding_mask: The padding mask for the question and answer, defaults to None
        :param padding_mask: tf.Tensor, optional
        :param shift_target_sequence_right: Shift the target sequence to the right, defaults to True
        :param shift_target_sequence_right: bool, optional
        :param training: Define the training variables of the network, defaults to True
        :param training: bool, optional
        :return: Output logits of the model
        :rtype: tf.Tensor
        """

        context, question, context_characters, question_characters, _, _, _ = inputs

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

        # TODO: Compute something useful with the answer indices
        # answer_index1 = answer_index1[:, :context_maxlen]
        # answer_index2 = answer_index2[:, :context_maxlen]

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
        assert (mask.shape[0] == inputs.shape[0]) in [None, True], 'Mask and input batch size must match'
        assert len(mask.shape) == 2, 'Can only convert dimension 2 masks to dimension 3 masks'

        mask = tf.tile(mask[:, None, :], (1, tf.shape(inputs)[1], 1))
        return mask

    def _convert_seqlens_to_attention_mask(self, inputs, seqlens):
        assert (seqlens.shape[0] == inputs.shape[0]) in [None, True], 'Seqlens and input batch size must match'
        assert len(seqlens.shape) == 1, 'Can only convert dimension 1 seqlens to dimension 3 masks'

        indices = tf.tile(tf.range(tf.shape(inputs)[1])[
                          None, :], (tf.shape(seqlens)[0], 1))
        mask = indices < seqlens[:, None]
        return mask

    def _convert_padding_masks_to_context_query_mask(self, query_mask, context_mask):
        return tf.logical_and(context_mask[:, :, None], query_mask[:, None, :])
