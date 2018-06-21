import tensorflow as tf

from rl_algs.eager.common.layers import Residual, Stack, DenseStack
from rl_algs.eager.common.attention import SelfAttention, MultiHeadAttention

class TransformerEncoderBlock(tf.keras.Model):
	"""An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

	def __init__(self, n_heads, filter_size, hidden_size):
		super().__init__()
		attention = SelfAttention(n_heads)
		self.residual_attention = Residual(attention, norm=True)

		dense_relu_dense = DenseStack([filter_size, hidden_size], output_activation=None)
		self.residual_dense = Residual(dense_relu_dense, norm=True)

	def call(self, inputs, attention_mask=None):

		res_attn = self.residual_attention(inputs, attention_mask)

		output = self.residual_dense(res_attn)
		return output

class TransformerDecoderBlock(tf.keras.Model):

	def __init__(self, n_heads, filter_size, hidden_size):
		super().__init__()
		self_attention = SelfAttention(n_heads)
		self.residual_self_attention= Residual(self_attention, norm=True)

		multihead_attn = MultiHeadAttention(n_heads)
		self.residual_multi_attention = Residual(multihead_attn, norm=True)

		dense_relu_dense = DenseStack([filter_size, hidden_size], output_activation=None)
		self.residual_dense = Residual(dense_relu_dense, norm=True)

	def call(self, inputs, mask=None):
		decoder_inputs, encoder_outputs = inputs

		target_attention_out = self.residual_self_attention(decoder_inputs, mask)

		encdec_attention_out = self.residual_multi_attention((target_attention_out, encoder_outputs), mask)

		output = self.residual_dense()
