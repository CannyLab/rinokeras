import tensorflow as tf
from tensorflow.python.keras import backend as K  # pylint: disable=E0611


class LuongAttention(tf.keras.layers.Layer):

    def __init__(self, local=False, stddev=1.0):
        super().__init__()
        self.local = local
        if self.local:
            self.stddev = stddev

    def build(self, inputs):
        self.attention_weights = self.add_variable('attention_weights',
                                                   (inputs[0][-1] + inputs[1]
                                                    [-1], inputs[1][-1]),
                                                   initializer=tf.initializers.variance_scaling())

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
            position_weighting = tf.exp(-1. * tf.square(relative_position) /
                                        (2 * tf.square(self.stddev)))
            weights = alignment * tf.reshape(position_weighting, (1, -1, 1))

        # will broadcast over third dimension
        weighted = tf.reduce_sum(source_hidden_sequence * weights, 1)
        concatenated = tf.concat((target_hidden, weighted), 1)
        output = tf.tanh(tf.matmul(concatenated, self.attention_weights))
        return output

# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py


class AttentionQKV(tf.keras.Model):
    """Computes query, key, and value from antecedents

            :param key_depth: integer depth of query and keys
            :param value_depth: integer depth of values
    """

    def __init__(self,
                 key_depth: int,
                 value_depth: int = None) -> None:
        super().__init__()
        if value_depth is None:
            value_depth = key_depth

        # TODO: Splitting this up as three layers is slower than concatenating the layers,
        #       doing the transformation, and then re-separating the layers would be.
        self.query_layer = tf.keras.layers.Dense(key_depth, use_bias=False)
        self.key_layer = tf.keras.layers.Dense(key_depth, use_bias=False)
        self.value_layer = tf.keras.layers.Dense(value_depth, use_bias=False)

    def call(self, inputs):
        """
            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        query_antecedent, memory_antecedent = inputs
        queries = self.query_layer(query_antecedent)
        keys = self.key_layer(memory_antecedent)
        values = self.value_layer(memory_antecedent)

        return [queries, keys, values]


class TrilinearSimilarity(tf.keras.layers.Layer):
    """
    Computes Trilinear similarity between query and context tensors.

    Based on https://arxiv.org/pdf/1611.01603.pdf.
    """

    def __init__(self, dropout=None):
        super().__init__()
        self.dropout = None if dropout is None else tf.keras.layers.Dropout(
            dropout)

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
                                             initializer=tf.keras.initializers.glorot_uniform())
        self.context_weights = self.add_weight('context_weights',
                                               shape=(context_channels, 1),
                                               initializer=tf.keras.initializers.glorot_uniform())
        self.dot_weights = self.add_weight('dot_weights',
                                           shape=(context_channels,
                                                  context_channels),
                                           initializer=tf.keras.initializers.glorot_uniform())

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
        if self.dropout:
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


class ScaledDotProductSimilarity(tf.keras.layers.Layer):
    """
    Fast scaled dot product attention.

    Based on https://arxiv.org/abs/1706.03762.
    """

    def call(self, inputs):
        """
            Args:
                (query, keys) ->
                      query: a Tensor with shape [batch_size, heads (optional), query_length, channels]
                      keys: a Tensor with shape [batch_size, heads (optional), key_length, channels]

            Returns:
                similarity: a Tensor with shape [batch_size, heads (optional), query_length, key_length]
        """
        queries, keys = inputs
        key_dim = tf.cast(tf.shape(keys)[-1], tf.float32)

        similarity = tf.matmul(
            queries, keys, transpose_b=True) / tf.sqrt(key_dim)

        return similarity


class ApplyAttentionMask(tf.keras.layers.Layer):
    """
    Applies a mask to the attention similarities.
    """

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

        assert similarity.ndim in [
            3, 4], 'similarity must be a 3 or 4 dimensional Tensor'
        assert mask.ndim == 3, 'Mask must be a 3 dimensional Tensor'

        # There are so many different reasons a mask might be constructed a particular manner.
        # Because of this we don't want to infer a particular construction.
        assert mask.shape[0] == similarity.shape[0], 'Batch size mismatch between mask and similarity'
        assert mask.shape[-2] == similarity.shape[-2], 'Mismatch in dimension -2 between mask and similarity'
        assert mask.shape[-1] == similarity.shape[-1], 'Mismatch in dimension -1 between mask and similarity'

        if mask.ndim != similarity.ndim:
            mask = tf.expand_dims(mask, 1)

        bias = -1e9 * tf.cast(tf.logical_not(mask), tf.float32)
        masked_similarity = similarity + bias
        return masked_similarity


class AttentionMap(tf.keras.Model):
    """
    Computes attention based on provided similarity metric.
    """

    def __init__(self, similarity_metric, dropout: float = None) -> None:
        super().__init__()
        self.similarity_metric = similarity_metric
        self.apply_mask = ApplyAttentionMask()
        self.dropout = None if dropout is None else tf.keras.layers.Dropout(
            dropout)

    def call(self, inputs, mask=None):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_queries]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        queries, keys, values = inputs
        similarity = self.similarity_metric((queries, keys))
        masked_similarity = self.apply_mask(similarity, mask=mask)
        # (batch_size, heads, n_queries, n_keyval)
        weights = tf.nn.softmax(masked_similarity, axis=-1)

        if self.dropout is not None:
            weights = self.dropout(weights)
        output = tf.matmul(weights, values)
        return output


class MultiHeadAttentionMap(AttentionMap):

    def __init__(self, similarity_metric, n_heads: int, dropout: float = None) -> None:
        """Map the multi-headed attention across the map

        Arguments:
            similarity_metric {[type]} -- The metric that should be used for the similarity
            n_heads {int} -- The number of heads in the attention map

        Keyword Arguments:
            dropout {float} -- Dropout parameter (default: {None})
        """

        super().__init__(similarity_metric, dropout)
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
        attention_output_split = super().call(
            (queries_split, keys_split, values_split), mask=mask)
        output = self._combine_heads(attention_output_split)
        return output

    def _split_heads(self, tensor):
        assert len(tensor.shape) == 3, 'Tensor dimension invalid. Expected 3, Received {}'.format(
            len(tensor.shape))
        batch_size, tensorlen = tf.shape(tensor)[0], tf.shape(tensor)[1]
        feature_size = tensor.shape.as_list()[2]
        new_feature_size = feature_size // self.n_heads
        tensor = tf.reshape(tensor, (batch_size, tensorlen,
                                     self.n_heads, new_feature_size))
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        return tensor

    def _combine_heads(self, tensor):
        assert len(tensor.shape) == 4, 'Tensor dimension invalid. Expected 4, Received {}'.format(
            len(tensor.shape))
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        batch_size, tensorlen = tf.shape(tensor)[0], tf.shape(tensor)[1]
        feature_size = tensor.shape.as_list()[-1]
        new_feature_size = self.n_heads * feature_size
        tensor = tf.reshape(tensor, (batch_size, tensorlen, new_feature_size))
        return tensor


class MultiHeadAttention(tf.keras.Model):
    """
    Fast multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 similarity_metric: str,
                 n_heads: int,
                 dropout: float = None) -> None:
        super().__init__()
        if similarity_metric != "scaled_dot":
            raise NotImplementedError(
                "Haven't got around to implementing other attention types yet!")

        self.similarity_metric = similarity_metric
        self.n_heads = n_heads

        similarity_metric = ScaledDotProductSimilarity()
        self.attention_layer = MultiHeadAttentionMap(
            similarity_metric, n_heads, dropout)

        self.dropout = None if dropout is None else tf.keras.layers.Dropout(
            dropout)

    def build(self, input_shapes):
        query_antecedent_shape, memory_antecedent_shape = input_shapes
        qa_channels = query_antecedent_shape[-1]
        ma_channels = memory_antecedent_shape[-1]
        assert qa_channels % self.n_heads == 0 and ma_channels % self.n_heads == 0, \
            'Feature size must be divisible by n_heads'
        assert qa_channels == ma_channels, 'Cannot combine tensors with different shapes'
        self.compute_qkv = AttentionQKV(qa_channels,
                                        ma_channels)
        self.output_layer = tf.keras.layers.Dense(ma_channels, use_bias=False)

    def call(self, inputs, mask=None):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        query_antecedent, memory_antecedent = inputs
        q, k, v = self.compute_qkv((query_antecedent, memory_antecedent))
        attention_output = self.attention_layer((q, k, v), mask=mask)
        output = self.output_layer(attention_output)
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class SelfAttention(MultiHeadAttention):
    """
    Fast multi-head self attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 similarity_metric: str,
                 n_heads: int,
                 dropout: float = None) -> None:
        super().__init__(similarity_metric, n_heads, dropout)

    def build(self, input_shapes):
        super().build((input_shapes, input_shapes))

    def call(self, inputs, mask=None):
        return super().call((inputs, inputs), mask=mask)


class ContextQueryAttention(tf.keras.Model):

    def __init__(self, n_heads, output_depth, attention_type="trilinear", dropout=None):
        super().__init__()
        if attention_type != "trilinear":
            raise NotImplementedError(
                "Haven't got around to implementing other attention types yet!")

        self.n_heads = n_heads
        self.attention_type = attention_type
        self.dropout = dropout
        self.apply_mask = ApplyAttentionMask()
        self.trilinear_similarity = TrilinearSimilarity(output_depth)

    def call(self, inputs, mask=None):
        """
            Args:
                (query, context) ->
                      query: a Tensor with shape [batch_size, query_length, d_model]
                      context: a Tensor with shape [batch_size, context_length, d_model]

            Returns:
                outputs: a Tensor with shape [batch_size, context_length, 4 * d_model]
        """
        query, context = inputs

        # similarity -> Tensor with shape [batch_size, context_length, query_length]
        similarity = self.trilinear_similarity((query, context))
        masked_similarity = self.apply_mask(similarity, mask=mask)

        c2q_similarity = tf.nn.softmax(masked_similarity, axis=-1)
        q2c_similarity = tf.nn.softmax(masked_similarity, axis=-2)

        # context_to_query -> Tensor with shape [batch_size, context_length, d_model]
        context_to_query = tf.matmul(c2q_similarity, query)
        # query_to_context -> Tensor with shape [batch_size, context_length, d_model]
        query_to_context = tf.matmul(
            tf.matmul(c2q_similarity, q2c_similarity, transpose_b=True), context)

        # outputs -> Tensor with shape [batch_size, context_length, 4 * d_model]
        outputs = [context, context_to_query, context *
                   context_to_query, context * query_to_context]
        return tf.concat(outputs, axis=-1)
