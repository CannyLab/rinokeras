from typing import Optional, Callable, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dropout
import tensorflow.keras.backend as K  # pylint: disable=E0611

from .layers import WeightNormDense as Dense
from .layers import LayerNorm

class LuongAttention(Layer):

    def __init__(self, local=False, stddev=1.0, regularizer=None):
        super().__init__()
        self.local = local
        self.regularizer = regularizer
        if self.local:
            self.stddev = stddev

    def build(self, input_shape):
        if self.local:
            inputs0, inputs1, _ = input_shape
        else:
            inputs0, inputs1 = input_shape
        self.attention_weights = self.add_variable('attention_weights',
                                                (int(inputs0[-1]) + int(inputs1[-1]), int(inputs1[-1])),
                                                initializer=tf.initializers.variance_scaling(),
                                                regularizer=self.regularizer)
        super().build(input_shape)


    def call(self, inputs):
        if self.local:
            target_hidden, source_hidden_sequence, positions = inputs
        else:
           target_hidden, source_hidden_sequence = inputs
        # source hidden sequence shape -> (None, None, encoder_cell_size)
        # target hidden shape -> (None, decoder_cell_size)
        score = tf.matmul(source_hidden_sequence,
                          tf.expand_dims(target_hidden, -1))
        alignment = tf.nn.softmax(score, 1)
        weights = alignment

        if self.local:
            relative_position = tf.cast(tf.range(source_hidden_sequence.shape[1]), positions.dtype) -  tf.reshape(positions, (-1, 1))
            position_weighting = tf.exp(tf.cast(-1.,source_hidden_sequence.dtype) * tf.cast(tf.square(relative_position),source_hidden_sequence.dtype) / \
                tf.cast((2 * tf.square(self.stddev)), source_hidden_sequence.dtype))
            weights = alignment * position_weighting[:,:,None]

        # will broadcast over third dimension
        weighted = tf.reduce_sum(source_hidden_sequence * weights, 1)
        concatenated = tf.concat((target_hidden, weighted), 1)
        output = tf.tanh(tf.matmul(concatenated, self.attention_weights))
        return output

# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

class AttentionQKVProjection(Model):
    """Computes query, key, and value from antecedents

            :param key_depth: integer depth of query and keys
            :param value_depth: integer depth of values
    """

    def __init__(self,
                 key_depth: int,
                 value_depth: int = None,
                 project_value: bool = True,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        if value_depth is None:
            value_depth = key_depth

        self.key_depth = key_depth
        self.value_depth = value_depth
        self.project_value = project_value

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.query_layer = Dense(self.key_depth, use_bias=False,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 bias_regularizer=self.bias_regularizer,
                                 activity_regularizer=self.activity_regularizer)
        self.query_norm = LayerNorm()
        self.key_layer = Dense(self.key_depth, use_bias=False,
                                      kernel_initializer=kernel_initializer,
                                      kernel_regularizer=self.kernel_regularizer,
                                      bias_regularizer=self.bias_regularizer,
                                      activity_regularizer=self.activity_regularizer)
        self.key_norm = LayerNorm()

        if self.project_value:
            self.value_layer = Dense(self.value_depth, use_bias=False,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        activity_regularizer=self.activity_regularizer)
            self.value_norm = LayerNorm()

    def call(self, inputs):
        """
            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                key_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                value_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
        """
        query_antecedent, key_antecedent, value_antecedent = inputs
        queries = self.query_norm(self.query_layer(query_antecedent))
        keys = self.key_norm(self.key_layer(key_antecedent))
        if self.project_value:
            values = self.value_norm(self.value_layer(value_antecedent))
        else:
            values = value_antecedent
        return [queries, keys, values]

class TrilinearSimilarity(Layer):
    """
    Computes Trilinear similarity between query and context tensors.

    Based on https://arxiv.org/pdf/1611.01603.pdf.
    """

    def __init__(self,
                 dropout: Optional[float] = None,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 regularizer=None) -> None:
        super().__init__()
        self.dropout = Dropout(0 if dropout is None else dropout)
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularizer

    def build(self, input_shapes):
        """
            Args:
                (query_shape, context_shape) ->
                    context_shape: a tf.Shape [batch_size, context_length, channels]
                    query_shape: a tf.Shape [batch_size, query_length, channels]

            Returns: None
        """
        context_shape, query_shape = input_shapes
        query_channels = query_shape.as_list()[-1]
        context_channels = context_shape.as_list()[-1]

        self.query_weights = self.add_weight('query_weights',
                                             shape=(query_channels, 1),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.regularizer)
        self.context_weights = self.add_weight('context_weights',
                                               shape=(context_channels, 1),
                                               initializer=self.kernel_initializer,
                                               regularizer=self.regularizer)
        self.dot_weights = self.add_weight('dot_weights',
                                           shape=(context_channels,
                                                  context_channels),
                                           initializer=self.kernel_initializer,
                                           regularizer=self.regularizer)
        super().build(input_shapes)

    def call(self, inputs):
        """
            Args:
                (query, context) ->
                    context: a Tensor with shape [batch_size, context_length, channels]
                    query: a Tensor with shape [batch_size, query_length, channels]
            Returns:
                similarity: a Tensor with shape [batch_size, context_length, query_length]
        """
        context, query = inputs
        query = self.dropout(query)
        context = self.dropout(context)

        # context_weighted -> Tensor with shape [batch_size, context_length, 1]
        context_weighted = K.dot(context, self.context_weights)

        # query_weighted -> Tensor with shape [batch_size, 1, query_length]
        if len(query.shape) == 3:
            query_weighted = tf.transpose(
                K.dot(query, self.query_weights), (0, 2, 1))
        elif len(query.shape) == 4:
            # We need to account for the head dimension in multi-headed attention layers
            query_weighted = tf.transpose(
                K.dot(query, self.query_weights), (0, 1, 3, 2))
        else:
            raise NotImplementedError('Trilinear similairty only supported with 3 or 4 dimensions')

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
        super().__init__()

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
        key_dim = tf.cast(tf.shape(keys)[-1], queries.dtype)
        return tf.matmul(queries / tf.sqrt(key_dim), keys, transpose_b=True)

class ApplyAttentionMask(Layer):
    """
    Applies a mask to the attention similarities.
    """
    def __init__(self, hadamard=False):
        self.hadamard = hadamard
        super().__init__()

    def call(self, inputs, mask=None):
        """
            Args:
                  inputs: a Tensor with shape [batch_size, heads (optional), q/k_length, q/k_length]
                  mask: a Tensor with shape [batch_size, q/k_length, q/k_length]

            Returns:
                masked_similarity: a Tensor with shape [batch_size, heads (optional), q/k_length, q/k_length]
        """
        similarity = inputs
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
            if self.hadamard:
                # If we're not going through a softmax layer, apply the hadamard product between the mask and the similarity
                return tf.cast(tf.cast(mask, tf.bool), similarity.dtype) * similarity 
            bias = -1e9 * tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), similarity.dtype)
            masked_similarity = similarity + bias
            return masked_similarity

class AttentionMap(Model):
    """
    Computes attention based on provided similarity metric.
    """

    def __init__(self,
                 similarity_metric: Callable[[Tuple[tf.Tensor, tf.Tensor]], tf.Tensor],
                 attention_function: Callable[[tf.Tensor], tf.Tensor] = tf.nn.softmax,
                 dropout: Optional[float] = None) -> None:
        super().__init__()
        self.similarity_metric = similarity_metric
        self.attention_function = attention_function

        use_hadamard_mask = False if self.attention_function == tf.nn.softmax else True
        self.apply_mask = ApplyAttentionMask(hadamard=use_hadamard_mask)
        self.dropout = Dropout(0 if dropout is None else dropout)

    def call(self, inputs, mask=None, return_attention_weights=True):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_keyval]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        queries, keys, values = inputs
        similarity = self.similarity_metric((queries, keys))
        masked_similarity = self.apply_mask(similarity, mask=mask)
        # (batch_size, heads, n_queries, n_keyval)
        if self.attention_function == tf.nn.softmax:
            weights = self.attention_function(masked_similarity - tf.reduce_max(masked_similarity, -1, keepdims=True))
        else:
            weights = self.attention_function(masked_similarity)
        weights = self.dropout(weights)
        tf.add_to_collection('ATTENTION_WEIGHTS', weights)
        output = tf.matmul(weights, values)
        if return_attention_weights:
            return output, weights
        return output

class MultiHeadAttentionMap(Model):

    def __init__(self,
                 similarity_metric: Callable[[Tuple[tf.Tensor, tf.Tensor]], tf.Tensor],
                 n_heads: int,
                 attention_function: Callable[[tf.Tensor], tf.Tensor] = tf.nn.softmax,
                 dropout: Optional[float] = None) -> None:
        """Map the multi-headed attention across the map

        Arguments:
            similarity_metric {[type]} -- The metric that should be used for the similarity
            n_heads {int} -- The number of heads in the attention map

        Keyword Arguments:
            dropout {float} -- Dropout parameter (default: {None})
        """

        super().__init__()
        self.attention_map = AttentionMap(similarity_metric, attention_function=attention_function, dropout=dropout)
        self.n_heads = n_heads

    def build(self, input_shape):
        for shape in input_shape:
            assert shape[-1] % self.n_heads == 0, 'Shape of feature input must be divisible by n_heads'

    def call(self, inputs, mask=None, return_attention_weights=False):
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
        attention_output_split, attention_weights = self.attention_map(
            (queries_split, keys_split, values_split), mask=mask)
        output = self._combine_heads(attention_output_split)
        if return_attention_weights:
            return output, attention_weights
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
                 key_size: Optional[int] = None,
                 value_size: Optional[int] = None,
                 attention_function: Callable[[tf.Tensor], tf.Tensor] = tf.nn.softmax,
                 project_value: bool = True, 
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer]=None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer]=None,
                 activity_regularizer: Optional[tf.keras.regularizers.Regularizer]=None) -> None:
        super().__init__()

        # Setup the similarity map
        sm_map = {
            'scaled_dot': ScaledDotProductSimilarity,
            'trilinear': TrilinearSimilarity,
        }
        if similarity_metric in sm_map:
            self.similarity_metric_str = similarity_metric
            self.similarity_metric = sm_map[similarity_metric]()
        else:
            raise NotImplementedError('Similarity metric {} is not implemented. Choose one of {}'.format(similarity_metric, sm_map.keys()))

        self.key_size = key_size
        self.value_size = value_size
        self.n_heads = n_heads
        self.project_value = project_value
        self.attention_function = attention_function
        assert key_size is None or key_size % n_heads == 0, \
            'Key size must be divisible by n_heads if provided'

        self.attention_layer = MultiHeadAttentionMap(
            self.similarity_metric, n_heads, self.attention_function, dropout)

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.dropout_pct = dropout
        self.dropout = Dropout(0 if dropout is None else dropout)

    def build(self, input_shapes):
        query_antecedent_shape, key_antecedent_shape, value_antecedent_shape = input_shapes
        self.query_size = query_antecedent_shape[-1] if self.key_size is None else self.key_size
        self.key_size = key_antecedent_shape[-1] if self.key_size is None else self.key_size
        self.value_size = value_antecedent_shape[-1] if self.value_size is None else self.value_size

        assert self.key_size == self.query_size
        assert self.key_size % self.n_heads == 0, 'Feature size must be divisible by n_heads'
        # assert qa_channels == ma_channels, 'Cannot combine tensors with different shapes'
        self.compute_qkv = AttentionQKVProjection(self.key_size, self.value_size, self.project_value,
                                        kernel_initializer=self.kernel_initializer,
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        activity_regularizer=self.activity_regularizer)
        self.output_layer = Dense(self.value_size, use_bias=False,
                                  kernel_initializer=self.kernel_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  activity_regularizer=self.activity_regularizer)

    def call(self, inputs, mask=None, return_attention_weights=False):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, key_antecedent, value_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                key_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
                value_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        assert isinstance(inputs, tuple) or isinstance(inputs, list) and len(inputs) == 2, \
            'Must pass query and memory'
        qa, ma, va = inputs
        qkv_projection = self.compute_qkv((qa, ma, va))
        attention_output, attention_weights = self.attention_layer(
            qkv_projection, mask=mask, return_attention_weights=True)
        
        output = self.output_layer(attention_output)
        output = self.dropout(output)
        if return_attention_weights:
            return output, attention_weights
        return output

    def get_base_config(self):
        config = {
            'similarity_metric': self.similarity_metric_str,
            'n_heads': self.n_heads,
            'dropout': self.dropout_pct,
            'key_size': self.key_size,
            'value_size': self.value_size,
            'attention_function': tf.keras.utils.serialize_keras_object(self.attention_function),
            'project_value': self.project_value,
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
    
    def get_config(self):
        config = self.get_base_config()
        return config
        # base_config = super(MultiHeadAttention, self).get_config()
        # return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
        super().__init__()
        self.multi_attention = MultiHeadAttention(similarity_metric=similarity_metric, n_heads=n_heads, dropout=dropout, **kwargs)

    def call(self, inputs, mask=None, return_attention_weights=False):
        return self.multi_attention((inputs, inputs, inputs), mask=mask, return_attention_weights=return_attention_weights)

    def get_config(self):
        config = self.multi_attention.get_base_config()
        return config
        # base_config = super(SelfAttention, self).get_config()
        # return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ContextQueryAttention(Model):

    def __init__(self,
                 similarity_metric: str = "trilinear",
                 dropout: Optional[float] = None,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 regularizer=None) -> None:
        super().__init__()
        if similarity_metric != "trilinear":
            raise NotImplementedError(
                "Haven't got around to implementing other attention types yet!")

        self.similarity_metric = similarity_metric
        self.dropout = Dropout(0 if dropout is None else dropout)
        self.apply_mask = ApplyAttentionMask()
        self.trilinear_similarity = TrilinearSimilarity(
            dropout,
            kernel_initializer=kernel_initializer,
            regularizer=regularizer)

    def call(self, inputs, mask=None):
        """
        Args:
            (query, context) ->
                  query: a Tensor with shape [batch_size, query_length, d_model]
                  context: a Tensor with shape [batch_size, context_length, d_model]

        Returns:
            outputs: a Tensor with shape [batch_size, context_length, 4 * d_model]
        """
        context, query = inputs
        # similarity -> Tensor with shape [batch_size, context_length, query_length]
        similarity = self.trilinear_similarity((context, query))
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
