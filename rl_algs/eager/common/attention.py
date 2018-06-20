import collections
import tensorflow as tf
import numpy as np

class LuongAttention(tf.keras.layers.Layer):

    def __init__(self, local=False, stddev=1.0):
        super().__init__()
        self.local = local
        if self.local:
            self.stddev = stddev

    def build(self, inputs):
        self.attention_weights = self.add_variable('attention_weights', (inputs[0][-1] + inputs[1][-1], inputs[1][-1]),
                                                                        initializer=tf.initializers.variance_scaling())

    def call(self, inputs, t=None):
        target_hidden, source_hidden_sequence = inputs
        # source hidden sequence shape -> (None, None, encoder_cell_size)
        # target hidden shape -> (None, decoder_cell_size)
        score = tf.matmul(source_hidden_sequence, tf.expand_dims(target_hidden, -1))
        alignment = tf.nn.softmax(score, 1)
        weights = alignment

        if self.local:
            if t is None:
                raise TypeError("Must pass in position for local attention")
            relative_position = tf.cast(tf.range(source_hidden_sequence.shape[1]), tf.float32) - t
            position_weighting = tf.exp( - tf.square(relative_position) / (2 * tf.square(self.stddev)) )
            weights = alignment * tf.reshape(position_weighting, (1, -1, 1))
            
        weighted = tf.reduce_sum(source_hidden_sequence * weights, 1) # will broadcast over third dimension
        concatenated = tf.concat((target_hidden, weighted), 1)
        output = tf.tanh(tf.matmul(concatenated, self.attention_weights))
        return output

# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
class AttentionQKV(tf.keras.Model):

    def __init__(self, 
                 key_depth, 
                 value_depth, 
                 q_filter_width=1,
                 kv_filter_width=1,
                 q_padding="valid",
                 kv_padding="valid"):

        """Computes query, key, and valud from antecedents

            :param key_depth: integer depth of query and keys
            :param value_depth: integer depth of values
            :param q_filter_width: integer specifying how wide query should be
            :param kv_filter_width: integer specifying how wide you want the keys and values
            :param q_padding: padding for conv if filter width > 1
            :param kv_padding: padding for conv if filter width > 1
        """
        super().__init__()
        self.key_depth = key_depth
        self.value_depth = value_depth
        self.q_filter_width = q_filter_width
        self.kv_filter_width = kv_filter_width
        self.q_padding = q_padding
        self.kv_padding = kv_padding

        self.query_layer = self.get_layer_type(key_depth, q_filter_width, q_padding)
        self.key_layer = self.get_layer_type(key_depth, kv_filter_width, kv_padding)
        self.value_layer = self.get_layer_type(value_depth, kv_filter_width, kv_padding)

    def get_layer_type(self, depth, filter_width, padding):
        if filter_width == 1:
            return tf.keras.layers.Dense(depth, use_bias=False)
        else:
            return tf.keras.layers.Conv1D(depth, filter_width, padding=padding)

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

        return (queries, keys, values)


class ScaledDotProductAttentionMap(tf.keras.layers.Layer):

    def build(self, input_shapes):
        shape_dims = input_shapes[0].ndims
        assert all(shape.ndims == shape_dims for shape in input_shapes), 'All inputs must have same number of dimensions'

        q_shape = input_shapes[0]
        k_shape = input_shapes[1]
        v_shape = input_shapes[2]

        if shape_dims == 4:
            heads = q_shape[1]
            shape = (1, heads, q_shape[-2], k_shape[-2])
        else:
            shape = (1, q_shape[-2], k_shape[-2])

        self.bias = self.add_variable('bias', shape, initializer=tf.zeros_initializer())
        super().build(input_shapes)

    def call(self, inputs):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        queries, keys, values = inputs
        logits = tf.matmul(queries, keys, transpose_b=True) # (batch_size, heads, n_queries, n_keyval)
        logits += self.bias
        dk = float(keys.shape[-1].value)
        weights = tf.nn.softmax(logits / np.sqrt(dk))# (batch_size, heads, n_queries, n_keyval)

        output = tf.matmul(weights, values)
        return output

class MultiHeadAttentionMap(tf.keras.Model):

    def __init__(self, n_heads, output_depth, attention_type="scaled_dot"):
        super().__init__()

        if attention_type != "scaled_dot":
            raise NotImplementedError("Haven't got around to implementing other attention types yet!")
        self.attention_type = attention_type
        self.attention_layer = ScaledDotProductAttentionMap()
        self.n_heads = n_heads
        self.output_layer = tf.keras.layers.Dense(output_depth, use_bias=False)

    def build(self, input_shape):
        for shape in input_shape:
            assert shape[-1] % self.n_heads == 0, 'Shape of feature input must be divisible by n_heads'

    def call(self, inputs):
        """Fast multi-head scaled dot product attention.

        :param queries: Tensor with shape [batch_size, n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, n_keyval, depth_v]
        :return: output: Tensor with shape [batch_size, n_querires, depth_v]
        """
        queries, keys, values = inputs

        queries_split = self.split_heads(queries)
        keys_split = self.split_heads(keys)
        values_split = self.split_heads(values)

        attention_output_split = self.attention_layer((queries_split, keys_split, values_split))
        attention_output = self.combine_heads(attention_output_split)
        output = self.output_layer(attention_output)
        return output


    def split_heads(self, tensor):
        new_feature_size = tensor.shape[-1] // self.n_heads
        tensor = tf.reshape(tensor, tensor.shape.as_list()[:-1] + [self.n_heads, new_feature_size])
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        return tensor

    def combine_heads(self, tensor):
        tensor = tf.transpose(tensor, (0, 2, 1, 3))
        new_feature_size = tensor.shape[-2] * tensor.shape[-1]
        tensor = tf.reshape(tensor, tensor.shape.as_list()[:-2] + [new_feature_size])
        return tensor

class SelfAttention(tf.keras.Model):

    def __init__(self, n_heads, attention_type="scaled_dot"):
        super().__init__()
        if attention_type != "scaled_dot":
            raise NotImplementedError("Haven't got around to implementing other attention types yet!")

        self.n_heads = n_heads
        self.attention_type = attention_type

    def build(self, input_shape):
        channels = input_shape[-1]
        assert channels % self.n_heads == 0, 'Feature size must be divisible by n_heads'
        self.compute_qkv = AttentionQKV(channels, channels)
        self.attention_layer = MultiHeadAttentionMap(self.n_heads, channels)

    def call(self, inputs):
        """Fast multi-head self attention.

        :param inputs: Tensor with shape [batch_size, sequence_length, channels]

        :return: output: Tensor with same shape as input
        """
        q, k, v = self.compute_qkv((inputs, inputs))
        return self.attention_layer((q, k, v))

class MultiHeadAttention(tf.keras.Model):

    def __init__(self, n_heads, attention_type="scaled_dot"):
        super().__init__()
        if attention_type != "scaled_dot":
            raise NotImplementedError("Haven't got around to implementing other attention types yet!")

        self.n_heads = n_heads
        self.attention_type = attention_type

    def build(self, input_shapes):
        query_antecedent_shape, memory_antecedent_shape = input_shapes
        qa_channels = query_antecedent_shape[-1]
        ma_channels = memory_antecedent_shape[-1]
        assert qa_channels % self.n_heads == 0 and ma_channels % self.n_heads == 0, 'Feature size must be divisible by n_heads'
        self.compute_qkv = AttentionQKV(qa_channels, ma_channels)
        self.attention_layer = MultiHeadAttentionMap(self.n_heads, ma_channels)

    def call(self, inputs):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        query_antecedent, memory_antecedent = inputs
        q, k, v = self.compute_qkv((query_antecedent, memory_antecedent))
        return self.attention_layer((q, k, v))