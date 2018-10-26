from typing import Optional, Callable

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Dropout
import tensorflow.keras.backend as K  # pylint: disable=E0611


class LuongAttention(Layer):

    def __init__(self, local=False, stddev=1.0, regularizer=None):
        super(LuongAttention, self).__init__()
        self.local = local
        self.regularizer = regularizer
        if self.local:
            self.stddev = stddev

    def build(self, input_shape):
        inputs0, inputs1 = input_shape
        self.attention_weights = self.add_variable('attention_weights',
                                                   (inputs0[-1] + inputs1
                                                    [-1], inputs1[-1]),
                                                   initializer=tf.initializers.variance_scaling(),
                                                   regularizer=self.regularizer)
        super().build(input_shape)

    def call(self, inputs, t=None):
        target_hidden, source_hidden_sequence = inputs
        # source hidden sequence shape -> (None, None, encoder_cell_size)
        # target hidden shape -> (None, decoder_cell_size)
        score = tf.matmul(source_hidden_sequence,
                          tf.expand_dims(target_hidden, -1))
        alignment = tf.nn.softmax(score, 1)
        weights = alignment

        if self.local:
            if t is None:
                raise TypeError("Must pass in position for local attention")
            relative_position = tf.cast(
                tf.range(source_hidden_sequence.shape[1]), tf.float32) - t
            position_weighting = tf.exp(-1. * tf.square(relative_position) / (2 * tf.square(self.stddev)))
            weights = alignment * tf.reshape(position_weighting, (1, -1, 1))

        # will broadcast over third dimension
        weighted = tf.reduce_sum(source_hidden_sequence * weights, 1)
        concatenated = tf.concat((target_hidden, weighted), 1)
        output = tf.tanh(tf.matmul(concatenated, self.attention_weights))
        return output

# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py


class AttentionQKV(Model):
    """Computes query, key, and value from antecedents

            :param key_depth: integer depth of query and keys
            :param value_depth: integer depth of values
    """

    def __init__(self,
                 key_depth: int,
                 value_depth: int = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(AttentionQKV, self).__init__()
        if value_depth is None:
            value_depth = key_depth

        self.key_depth = key_depth
        self.value_depth = value_depth

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        self.query_layer = Dense(self.key_depth, use_bias=False,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer,
                                 activity_regularizer=self.activity_regularizer)
        self.projection_layer = Dense(self.key_depth + self.value_depth, use_bias=False,
                                      kernel_regularizer=self.kernel_regularizer,
                                      bias_regularizer=self.bias_regularizer,
                                      activity_regularizer=self.activity_regularizer)

    def call(self, inputs):
        """
            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        query_antecedent, memory_antecedent = inputs
        queries = self.query_layer(query_antecedent)
        projection = self.projection_layer(memory_antecedent)
        keys, values = tf.split(projection, tf.stack((self.key_depth, self.value_depth)), axis=-1)

        return [queries, keys, values]


class TrilinearSimilarity(Layer):
    """
    Computes Trilinear similarity between query and context tensors.

    Based on https://arxiv.org/pdf/1611.01603.pdf.
    """

    def __init__(self, dropout: Optional[float] = None, regularizer=None) -> None:
        super().__init__()
        self.dropout = Dropout(0 if dropout is None else dropout)
        self.regularizer = regularizer

    def build(self, input_shapes):
        """
            Args:
                (query_shape, context_shape) ->
                      query_shape: a tf.Shape [batch_size, query_length, channels]
                      context_shape: a tf.Shape [batch_size, context_length, channels]

            Returns: None
        """
        query_shape, context_shape = input_shapes
        query_channels = query_shape.as_list()[-1]
        context_channels = context_shape.as_list()[-1]

        self.query_weights = self.add_weight('query_weights',
                                             shape=(query_channels, 1),
                                             initializer=tf.keras.initializers.glorot_uniform(),
                                             regularizer=self.regularizer)
        self.context_weights = self.add_weight('context_weights',
                                               shape=(context_channels, 1),
                                               initializer=tf.keras.initializers.glorot_uniform(),
                                               regularizer=self.regularizer)
        self.dot_weights = self.add_weight('dot_weights',
                                           shape=(context_channels,
                                                  context_channels),
                                           initializer=tf.keras.initializers.glorot_uniform(),
                                           regularizer=self.regularizer)
        super().build(input_shapes)

    def call(self, inputs):
        """
            Args:
                (query, context) ->
                      query: a Tensor with shape [batch_size, query_length, channels]
                      context: a Tensor with shape [batch_size, context_length, channels]

            Returns:
                similarity: a Tensor with shape [batch_size, context_length, query_length]
        """
        query, context = inputs
        query = self.dropout(query)
        context = self.dropout(context)

        # context_weighted -> Tensor with shape [batch_size, context_length, 1]
        context_weighted = K.dot(context, self.context_weights)

        # query_weighted -> Tensor with shape [batch_size, 1, query_length]
        query_weighted = tf.transpose(
            K.dot(query, self.query_weights), (0, 2, 1))

        # weighted_context_query -> Tensor with shape [batch_size, context_length, query_length]
        weighted_context_query = tf.matmul(
            K.dot(context, self.dot_weights), query, transpose_b=True)

        similarity = weighted_context_query + context_weighted + query_weighted
        return similarity


class ScaledDotProductSimilarity(Layer):
    """
    Fast scaled dot product attention.

    Based on https://arxiv.org/abs/1706.03762.
    """
    def __init__(self,):
        super(ScaledDotProductSimilarity, self).__init__()

    def call(self, queries, keys):
        """
            Args:
                (query, keys) ->
                      query: a Tensor with shape [batch_size, heads (optional), query_length, channels]
                      keys: a Tensor with shape [batch_size, heads (optional), key_length, channels]

            Returns:
                similarity: a Tensor with shape [batch_size, heads (optional), query_length, key_length]
        """
        key_dim = tf.cast(tf.shape(keys)[-1], tf.float32)

        similarity = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(key_dim)

        return similarity


class ApplyAttentionMask(Layer):
    """
    Applies a mask to the attention similarities.
    """
    def __init__(self, ):
        super(ApplyAttentionMask, self).__init__()

    def call(self, similarity, mask=None):
        """
            Args:
                  similarity: a Tensor with shape [batch_size, heads (optional), q/k_length, q/k_length]
                  mask: a Tensor with shape [batch_size, q/k_length, q/k_length]

            Returns:
                masked_similarity: a Tensor with shape [batch_size, heads (optional), q/k_length, q/k_length]
        """
        if mask is None:
            return similarity

        similarity_rank_assert = tf.assert_rank_in(similarity, (3, 4))
        mask_rank_assert = tf.assert_rank(mask, 3)

        # There are so many different reasons a mask might be constructed a particular manner.
        # Because of this we don't want to infer a particular construction.
        with tf.control_dependencies([similarity_rank_assert, mask_rank_assert]):
            # If shapes don't match, then similarity has been split for multi-headed attention
            if len(mask.shape) != len(similarity.shape):
                similarity[:, 0].shape.assert_is_compatible_with(mask.shape)
                mask = mask[:, None]
            else:
                similarity.shape.assert_is_compatible_with(mask.shape)

            # We know that we're passing this through a softmax later, thus just add a relatively large negative
            # value to mask the output avoids a hadamard product (though I think that technically it's not
            # any more efficient to do it this way operations wise)
            bias = -1e9 * tf.cast(tf.logical_not(mask), tf.float32)
            masked_similarity = similarity + bias
            return masked_similarity


class AttentionMap(Model):
    """
    Computes attention based on provided similarity metric.
    """

    def __init__(self,
                 similarity_metric,
                 attention_function: Callable[[tf.Tensor], tf.Tensor] = tf.nn.softmax,
                 dropout: Optional[float] = None) -> None:
        super(AttentionMap, self).__init__()
        self.similarity_metric = similarity_metric
        self.attention_function = attention_function
        self.apply_mask = ApplyAttentionMask()
        self.dropout = Dropout(0 if dropout is None else dropout)

    def call(self, queries, keys, values, mask=None):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_queries]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        similarity = self.similarity_metric(queries, keys)
        masked_similarity = self.apply_mask(similarity, mask=mask)
        # (batch_size, heads, n_queries, n_keyval)
        weights = self.attention_function(masked_similarity)

        weights = self.dropout(weights)
        output = tf.matmul(weights, values)
        return output, weights


class MultiHeadAttentionMap(Model):

    def __init__(self, similarity_metric, n_heads: int, dropout: Optional[float] = None) -> None:
        """Map the multi-headed attention across the map

        Arguments:
            similarity_metric {[type]} -- The metric that should be used for the similarity
            n_heads {int} -- The number of heads in the attention map

        Keyword Arguments:
            dropout {float} -- Dropout parameter (default: {None})
        """

        super(MultiHeadAttentionMap, self).__init__()
        self.attention_map = AttentionMap(similarity_metric, dropout=dropout)
        self.n_heads = n_heads

    def build(self, input_shape):
        for shape in input_shape:
            assert shape[-1] % self.n_heads == 0, 'Shape of feature input must be divisible by n_heads'

    def call(self, inputs, mask=None):
        """Fast multi-head attention.

        :param queries: Tensor with shape [batch_size, n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, n_keyval, depth_v]

        :return: output: Tensor with shape [batch_size, n_queries, depth_v]
        """
        queries, keys, values = inputs

        queries_split = self._split_heads(queries)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)
        attention_output_split, _ = self.attention_map(
            queries_split, keys_split, values_split, mask=mask)
        output = self._combine_heads(attention_output_split)
        return output

    def _split_heads(self, tensor):
        tensor.shape.assert_has_rank(3)
        batch_size, tensorlen = tf.shape(tensor)[0], tf.shape(tensor)[1]
        feature_size = tensor.shape.as_list()[2]
        new_feature_size = feature_size // self.n_heads
        tensor = tf.reshape(tensor, (batch_size, tensorlen,
                                     self.n_heads, new_feature_size))
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        return tensor

    def _combine_heads(self, tensor):
        tensor.shape.assert_has_rank(4)
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        batch_size, tensorlen = tf.shape(tensor)[0], tf.shape(tensor)[1]
        feature_size = tensor.shape.as_list()[-1]
        new_feature_size = self.n_heads * feature_size
        tensor = tf.reshape(tensor, (batch_size, tensorlen, new_feature_size))
        return tensor


class MultiHeadAttention(Model):
    """
    Fast multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 similarity_metric: str,
                 n_heads: int,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(MultiHeadAttention, self).__init__()
        if similarity_metric != "scaled_dot":
            raise NotImplementedError(
                "Haven't got around to implementing other attention types yet!")

        self.similarity_metric = similarity_metric
        self.n_heads = n_heads

        self.similarity_metric = ScaledDotProductSimilarity()
        self.attention_layer = MultiHeadAttentionMap(
            self.similarity_metric, n_heads, dropout)

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        self.dropout = Dropout(0 if dropout is None else dropout)

    def build(self, input_shapes):
        query_antecedent_shape, memory_antecedent_shape = input_shapes
        qa_channels = query_antecedent_shape[-1]
        ma_channels = memory_antecedent_shape[-1]
        assert qa_channels % self.n_heads == 0 and ma_channels % self.n_heads == 0, \
            'Feature size must be divisible by n_heads'
        assert qa_channels == ma_channels, 'Cannot combine tensors with different shapes'
        self.compute_qkv = AttentionQKV(qa_channels, ma_channels,
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        activity_regularizer=self.activity_regularizer)
        self.output_layer = Dense(ma_channels, use_bias=False,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  activity_regularizer=self.activity_regularizer)

    def call(self, inputs, mask=None):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        assert isinstance(inputs, tuple) or isinstance(inputs, list) and len(inputs) == 2, \
            'Must pass query and memory'
        q, k, v = self.compute_qkv(inputs)
        attention_output = self.attention_layer((q, k, v), mask=mask)
        output = self.output_layer(attention_output)
        output = self.dropout(output)
        return output


class SelfAttention(Model):
    """
    Fast multi-head self attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 similarity_metric: str,
                 n_heads: int,
                 dropout: Optional[float] = None,
                 **kwargs) -> None:
        super(SelfAttention, self).__init__()
        self.multi_attention = MultiHeadAttention(similarity_metric, n_heads, dropout, **kwargs)

    def call(self, inputs, mask=None):
        return self.multi_attention((inputs, inputs), mask=mask)


class ContextQueryAttention(Model):

    def __init__(self, attention_type: str = "trilinear", dropout: Optional[float] = None, regularizer=None) -> None:
        super(ContextQueryAttention, self).__init__()
        if attention_type != "trilinear":
            raise NotImplementedError(
                "Haven't got around to implementing other attention types yet!")

        self.attention_type = attention_type
        self.dropout = Dropout(0 if dropout is None else dropout)
        self.apply_mask = ApplyAttentionMask()
        self.trilinear_similarity = TrilinearSimilarity(dropout, regularizer=regularizer)

    def call(self, query, context, mask=None):
        """
        Args:
            (query, context) ->
                  query: a Tensor with shape [batch_size, query_length, d_model]
                  context: a Tensor with shape [batch_size, context_length, d_model]

        Returns:
            outputs: a Tensor with shape [batch_size, context_length, 4 * d_model]
        """

        # similarity -> Tensor with shape [batch_size, context_length, query_length]
        similarity = self.trilinear_similarity((query, context))
        masked_similarity = self.apply_mask(similarity, mask=mask)

        c2q_similarity = tf.nn.softmax(masked_similarity, axis=-1)
        q2c_similarity = tf.nn.softmax(masked_similarity, axis=-2)

        # context_to_query -> Tensor with shape [batch_size, context_length, d_model]
        context_to_query = tf.matmul(c2q_similarity, query)
        # query_to_context -> Tensor with shape [batch_size, context_length, d_model]
        query_to_context = tf.matmul(tf.matmul(c2q_similarity, q2c_similarity, transpose_b=True), context)

        # outputs -> Tensor with shape [batch_size, context_length, 4 * d_model]
        outputs = [context, context_to_query, context * context_to_query, context * query_to_context]
        outputs = tf.concat(outputs, axis=-1)
        outputs = self.dropout(outputs)
        return outputs


__all__ = ['LuongAttention', 'AttentionQKV', 'TrilinearSimilarity', 'ScaledDotProductSimilarity', 'ApplyAttentionMask',
           'AttentionMap', 'MultiHeadAttentionMap', 'MultiHeadAttention', 'SelfAttention', 'ContextQueryAttention']
