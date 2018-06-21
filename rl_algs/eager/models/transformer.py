import tensorflow as tf

from rl_algs.eager.common.layers import Residual, Stack, DenseStack
from rl_algs.eager.common.attention import SelfAttention, MultiHeadAttention

def get_position_encoding(max_seq_len, d_model):
	''' Sinusoid position encoding '''
	channels = np.arange(d_model)
	power = 2 * (channels // 2) / d_model
	divisor = np.power(10000, power)

	positions = np.arange(max_seq_len + 1) # plus one for padding

	position_encoding = positions[:,None] / divisor

	position_encoding[1:, ::2] = np.sin(position_encoding[1:, ::2]) # even dimensions
	position_encoding[1:, 1::2] = np.cos(position_encoding[1:, 1::2]) # odd dimensions
	return tf.constant(position_encoding, tf.float32)

class TransformerEncoderBlock(tf.keras.Model):
	"""An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

	def __init__(self, n_heads, filter_size, hidden_size, dropout=None):
		super().__init__()
		attention = SelfAttention(n_heads, dropout=dropout)
		self.residual_attention = Residual(attention, norm=True, dropout=dropout)

		dense_relu_dense = DenseStack([filter_size, hidden_size], output_activation=None)
		self.residual_dense = Residual(dense_relu_dense, norm=True, dropout=dropout)

	def call(self, inputs, attention_mask=None):

		res_attn = self.residual_attention(inputs, attention_mask)
		output = self.residual_dense(res_attn)
		return output

class TransformerDecoderBlock(tf.keras.Model):

	def __init__(self, n_heads, filter_size, hidden_size, dropout=None):
		super().__init__()
		self_attention = SelfAttention(n_heads, dropout=dropout)
		self.residual_self_attention= Residual(self_attention, norm=True, dropout=dropout)

		multihead_attn = MultiHeadAttention(n_heads, dropout=dropout)
		self.residual_multi_attention = Residual(multihead_attn, norm=True, dropout=dropout)

		dense_relu_dense = DenseStack([filter_size, hidden_size], output_activation=None)
		self.residual_dense = Residual(dense_relu_dense, norm=True, dropout=dropout)

	def call(self, inputs, self_attention_mask=None, attention_mask=None):
		decoder_inputs, encoder_outputs = inputs

		target_attention_out = self.residual_self_attention(decoder_inputs, mask=self_attention_mask)
		encdec_attention_out = self.residual_multi_attention((target_attention_out, encoder_outputs), mask=attention_mask)
		output = self.residual_dense(encdec_attention_out)
		return output

class TransformerEncoder(tf.keras.Model):

	def __init__(self, n_layers=6, n_heads=8, d_model=512, d_filter=2048, dropout=0.1):

		self.encoding_stack = Stack([TransformerEncoderBlock(n_heads, d_filter, d_model, dropout) for _ in range(n_layers)])