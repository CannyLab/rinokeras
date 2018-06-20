import collections
import tensorflow as tf

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


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def call(self, inputs):
        """Fast scaled dot product attention.

	        :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
	        :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
	        :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]

	        :return: output: Tensor with shape [batch_size, heads (optional), n_querires, depth_v]
        """
        queries, keys, values = inputs
        logits = tf.matmul(queries, keys, transpose_b=True) # (batch_size, heads, n_queries, n_keyval)
        dk = keys.shape[-1]
        weights = tf.nn.softmax(logits / tf.sqrt(dk))# (batch_size, heads, n_queries, n_keyval)

        output = tf.matmul(weights, values)
        return output

class MultiHeadAttention(tf.keras.Model):

    def __init__(self, n_heads, depth_k, depth_v, attention_type="scaled_dot"):
        super().__init__()
        if self.attention_type != "scaled_dot":
            raise NotImplementedError("Haven't got around to implementing other attention types yet!")
        self.attention_layer = ScaledDotProductAttention()
        self.query_heads = tf.keras.layers.Dense(n_heads * depth_k, use_bias=False)
        self.key_heads = tf.keras.layers.Dense(n_heads * depth_k, use_bias=False)
        self.value_heads = tf.keras.layers.Dense(n_heads * depth_v, use_bias=False)

    def call(self, inputs):
        """Fast multi-head scaled dot product attention.

        :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
        :return: output: Tensor with shape [batch_size, heads (optional), n_querires, depth_v]
        """
        