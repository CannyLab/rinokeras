import tensorflow as tf
import numpy as np

from rl_algs.eager.common.layers import Residual, Stack, DenseStack
from rl_algs.eager.common.attention import SelfAttention, MultiHeadAttention

class PositionEmbedding(tf.keras.Model):

    def __init__(self, hidden_size, discrete, n_symbols=None):
        super().__init__()
        assert hidden_size % 2 == 0, 'Hidden size must be even for sinusoidal encoding'
        self.discrete = discrete
        self.hidden_size = hidden_size
        self.n_symbols = n_symbols
        if discrete:
            assert n_symbols is not None and n_symbols > 0, 'Trying to predict a discrete value but received invalid n_symbols: {}'.format(n_symbols)
            self.input_embedding = tf.keras.layers.Embedding(n_symbols + 1, hidden_size, mask_zero=True)
        else:
            self.input_embedding = tf.keras.layers.Dense(hidden_size)

        power = tf.range(0, self.hidden_size, 2, dtype=tf.float32) / self.hidden_size
        divisor = 10000 ** power
        self._divisor = divisor

    def call(self, inputs, mask=None):
        """An embedding of the input. Input can be a discrete or continuous variable.

            :param inputs: Either an int32 Tensor with shape [batch_size, sequence_length] or a float32
                            Tensor with shape [batch_size, sequence_length, input_dim]
            :param mask: None if discrete, otherwise a boolean Tensor with shape [batch_size, sequence_length]

            :return embedding: float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        batch_size, sequence_length = tf.shape(inputs)[0], tf.shape(inputs)[1] 

        position_embedding = self.get_position_embedding(sequence_length) # return [1, sequence_length, hidden_size]
        embedding = self.input_embedding(inputs) # return [batch_size, sequence_length, hidden_size]
        
        embedding += position_embedding
        if mask is not None:
            embedding *= tf.cast(mask[:,:,None], tf.float32)
        return embedding

    def get_position_embedding(self, sequence_length):
        """Sinusoid position encoding

           Takes matrix of sequence positions and returns sinusoidal encoding. Sequence position "0"
           should be reserved for padding.

               :param sequence_length: an integer length of input sequence

               :return position_embedding: float32 Tensor with shape [1, sequence_length, hidden_size]
               
        """

        seq_pos = tf.range(sequence_length.value, dtype=tf.float32)[None, :] + 1
        embedding = seq_pos[:,:,None] / self._divisor

        sin_embedding = tf.sin(embedding)
        cos_embedding = tf.cos(embedding)
        position_embedding = tf.stack((sin_embedding, cos_embedding), -1)
        position_shape = (tf.shape(position_embedding)[0], tf.shape(position_embedding)[1], self.hidden_size)
        position_embedding = tf.reshape(position_embedding, position_shape)

        return position_embedding

class TransformerEncoderBlock(tf.keras.Model):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

    def __init__(self, n_heads, filter_size, hidden_size, dropout=None):
        super().__init__()
        attention = SelfAttention(n_heads, dropout=dropout)
        self.residual_self_attention = Residual(attention, norm=True, dropout=dropout)

        dense_relu_dense = DenseStack([filter_size, hidden_size], output_activation=None)
        self.residual_dense = Residual(dense_relu_dense, norm=True, dropout=dropout)

    def call(self, inputs, self_attention_mask=None):

        res_attn = self.residual_self_attention(inputs, mask=self_attention_mask)
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
        return output, encoder_outputs

class TransformerEncoder(tf.keras.Model):
    """Stack of TransformerEncoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention.
        Defaults to 6 layers of self-attention.
    """

    def __init__(self, discrete, n_symbols_in=None, n_layers=6, n_heads=8, d_model=512, d_filter=2048, dropout=0.1):
        super().__init__()
        self.input_embedding = PositionEmbedding(d_model, discrete, n_symbols_in)
        self.encoding_stack = Stack([TransformerEncoderBlock(n_heads, d_filter, d_model, dropout) for _ in range(n_layers)])

    # Mask here should just be a mask over padded positions
    def call(self, inputs, padding_mask=None):
        input_embedding = self.input_embedding(inputs, mask=padding_mask)
        output = self.encoding_stack(input_embedding, self_attention_mask=padding_mask)
        return output


class TransformerDecoder(tf.keras.Model):

    # TODO: Handle fast decoding at test time. Easiest way is probably to have a test boolean and change the query to only contain the last vector.
    #         Not sure about beam search, other methods of decoding for NLP.
    def __init__(self, discrete, n_symbols_out=None, n_layers=6, n_heads=8, d_model=512, d_filter=2048, dropout=0.1):
        super().__init__()
        self.target_embedding = PositionEmbedding(d_model, discrete, n_symbols_out)
        self.decoding_stack = Stack([TransformerDecoderBlock(n_heads, d_filter, d_model, dropout) for _ in range(n_layers)])

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def call(self, inputs, padding_mask=None, mask_future=False):
        targets, encoder_output = inputs
        target_embedding = self.target_embedding(targets, mask=padding_mask)
        
        batch_size = tf.shape(target_embedding)[0]
        timesteps = tf.shape(target_embedding)[1]
        future_and_padding_mask = self.get_future_mask(batch_size, timesteps, padding_mask) if mask_future else None

        output, _ = self.decoding_stack((target_embedding, encoder_output), self_attention_mask=future_and_padding_mask,
                                                                            attention_mask=padding_mask)
        return output

    def get_future_mask(self, batch_size, sequence_length, padding_mask):
        """Mask future targets and padding

            :param batch_size: a TF Dimension
            :param sequence_length: a TF Dimension
            :param padding_mask: None or bool Tensor with shape [batch_size, sequence_length]

            :return mask: bool Tensor with shape [batch_size, sequence_length, sequence_length]
        """

        mask = np.zeros((sequence_length.value, sequence_length.value), np.bool)
        mask[np.triu_indices(sequence_length.value)] = True

        mask = tf.constant(mask, dtype=tf.bool)
        if padding_mask is not None:
            mask = tf.logical_and(padding_mask[:,:,None], mask[None, :, :])
            mask = tf.logical_and(mask, padding_mask[:,None,:])

        return mask

class Transformer(tf.keras.Model):

    def __init__(self, 
                 discrete, 
                 n_symbols_in=None,
                 n_symbols_out=None, 
                 out_size=None,
                 output_activation=None,
                 n_layers=6, 
                 n_heads=8, 
                 d_model=512, 
                 d_filter=2048, 
                 dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(discrete, n_symbols_in, n_layers, n_heads, d_model, d_filter, dropout)
        self.decoder = TransformerDecoder(discrete, n_symbols_out, n_layers, n_heads, d_model, d_filter, dropout)

        if not discrete:
            assert out_size is not None and out_size > 0, 'if not discrete, must specify output size'

        self.output_layer = tf.keras.layers.Dense(n_symbols_out if discrete else out_size, activation=output_activation)

    def fast_decode(self, inputs):
        raise NotImplementedError("Fast Decoding not implemented yet")

    def call(self, inputs, padding_mask=None, shift_target_sequence_right=True):
        source_sequence, target_sequence = inputs

        if shift_target_sequence_right:
            batch_size, target_size = tf.shape(target_sequence)[0], tf.shape(target_sequence)[2]
            first_zeros = tf.zeros((batch_size, 1, target_size))
            target_sequence = tf.concat((first_zeros, target_sequence[:,:-1]), axis=1)

        encoder_output = self.encoder(source_sequence, padding_mask=padding_mask)

        decoder_output = self.decoder((target_sequence, encoder_output), padding_mask=padding_mask, mask_future=True)

        output = self.output_layer(decoder_output)

        return output


