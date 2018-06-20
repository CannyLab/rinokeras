import tensorflow as tf

from rl_algs.eager.common.layers import Residual, Stack, DenseStack
from rl_algs.eager.common.attention import SelfAttention

class TransformerEncoderBlock(tf.keras.Model):
	"""An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

	def __init__(self, n_heads):
		super().__init__()
		attention = SelfAttention(n_heads)
		self.residual_attention = Residual(attention, norm=True)


	def build(self, input_shape):
		hidden_size = input_shape[-1]
		filter_size = hidden_size * 4

		dense_relu_dense = DenseStack([filter_size, hidden_size], output_activation=None)
		self.residual_dense = Residual(dense_relu_dense, norm=True)

	def call(self, inputs):

		res_attn = self.residual_attention(inputs)

		output = self.residual_dense(res_attn)
		return output