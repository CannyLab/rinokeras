from typing import Optional

import tensorflow as tf

from rl_algs.eager.common.layers import Residual, Stack, DenseStack, LayerNorm, PositionEmbedding
from rl_algs.eager.common.attention import MultiHeadAttention, SelfAttention


class TransformerSelfAttention(tf.keras.Model):

    def __init__(self, n_heads: int, dropout: Optional[float]) -> None:
        super().__init__()
        selfattn = SelfAttention('scaled_dot', n_heads, dropout)
        self.residual_self_attention = Residual(selfattn)
        self.norm = LayerNorm()

    def call(self, inputs, mask):
        resattn = self.residual_self_attention(inputs, mask=mask)
        return self.norm(resattn)


class TransformerMultiAttention(tf.keras.Model):
    def __init__(self, n_heads: int, dropout: Optional[float]) -> None:
        super().__init__()
        multiattn = MultiHeadAttention('scaled_dot', n_heads, dropout)
        self.residual_multi_attention = Residual(multiattn)
        self.norm = LayerNorm()

    def call(self, inputs, mask):
        resattn = self.residual_multi_attention(inputs, mask=mask)
        return self.norm(resattn)


class TransformerFeedForward(tf.keras.Model):
    def __init__(self, filter_size: int, hidden_size: int, dropout: Optional[float]) -> None:
        super().__init__()
        dense_relu_dense = DenseStack(
            [filter_size, hidden_size], output_activation=None)
        if dropout is not None:
            dropout = tf.keras.layers.Dropout(dropout)
            dense_relu_dense = Stack([dense_relu_dense, dropout])
        self.residual_dense = Residual(dense_relu_dense)
        self.norm = LayerNorm()

    def call(self, inputs):
        dense_out = self.residual_dense(inputs)
        return self.norm(dense_out)


class TransformerEncoderBlock(tf.keras.Model):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

    def __init__(self,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float] = None) -> None:
        super().__init__()
        self.self_attention = TransformerSelfAttention(n_heads, dropout)
        self.feed_forward = TransformerFeedForward(
            filter_size, hidden_size, dropout)

    def call(self, inputs, self_attention_mask=None):

        res_attn = self.self_attention(inputs, mask=self_attention_mask)
        output = self.feed_forward(res_attn)
        return output


class TransformerDecoderBlock(tf.keras.Model):
    """A decoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: two Tensors encoder_outputs, decoder_inputs
                    encoder_outputs -> a Tensor with shape [batch_size, sequence_length, channels]
                    decoder_inputs -> a Tensor with shape [batch_size, decoding_sequence_length, channels]

    :return: output: Tensor with same shape as decoder_inputs
    """

    def __init__(self,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float] = None) -> None:
        super().__init__()
        self.self_attention = TransformerSelfAttention(n_heads, dropout)
        self.multi_attention = TransformerMultiAttention(n_heads, dropout)
        self.feed_forward = TransformerFeedForward(
            filter_size, hidden_size, dropout)

    def call(self, inputs, self_attention_mask=None, attention_mask=None, fast_decode=False):
        encoder_outputs, decoder_inputs = inputs
        target_selfattn = self.self_attention(
            decoder_inputs, mask=self_attention_mask)
        encdec_attention = self.multi_attention(
            (target_selfattn, encoder_outputs), mask=attention_mask)
        output = self.feed_forward(encdec_attention)
        return [encoder_outputs, output]


class TransformerEncoder(tf.keras.Model):
    """
    Stack of TransformerEncoderBlocks. Performs repeated self-attention.
    """

    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 d_model: int,
                 d_filter: int,
                 dropout: Optional[float] = None) -> None:
        super().__init__()
        self.encoding_stack = Stack([TransformerEncoderBlock(n_heads, d_filter, d_model, dropout)
                                     for _ in range(n_layers)])

    def call(self, inputs, attention_mask=None):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, d_model]
                attention_mask: a boolean Tensor with shape [batch_size, sequence_length, sequence_length]
            Returns:
                output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        assert inputs.ndim == 3, 'Input dimension incorrect: {}'.format(
            inputs.shape)
        if attention_mask is not None:
            assert attention_mask.ndim == 3, 'Mask dimension incorrect: {}'.format(
                attention_mask.shape)
            assert attention_mask.shape[-1] == attention_mask.shape[-2], \
                'Last two mask dimensions must match {}'.format(
                    attention_mask.shape)
            assert attention_mask.shape[-2] == inputs.shape[1], 'Mask sequence dimensions must match'

        output = self.encoding_stack(
            inputs, self_attention_mask=attention_mask)
        return output


class TransformerDecoder(tf.keras.Model):
    """Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    # TODO: Not sure about beam search, other methods of decoding for NLP.
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 d_model: int,
                 d_filter: int,
                 dropout: Optional[float] = None) -> None:
        super().__init__()
        self.decoding_stack = Stack([TransformerDecoderBlock(n_heads, d_filter, d_model, dropout)
                                     for _ in range(n_layers)])

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def call(self, inputs, attention_mask=None, mask_future=False, fast_decode=False):
        """
            Args:
                inputs: a tuple of (encoder_output, target_embedding)
                    encoder_output: a float32 Tensor with shape [batch_size, sequence_length, d_model]
                    target_embedding: a float32 Tensor with shape [batch_size, target_length, d_model]
                attention_mask: a boolean Tensor with shape [batch_size, target_length, sequence_length]
                mask_future: a boolean for whether to mask future states in target self attention
                fast_decode: Not Implemented

            Returns:
                a tuple of (encoder_output, output)
                    output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        encoder_output, target_embedding = inputs

        assert target_embedding.ndim == 3, 'Target dimension incorrect: {}'.format(
            target_embedding.shape)
        assert encoder_output.ndim == 3, 'Encoder dimension incorrect: {}'.format(
            encoder_output.shape)
        if attention_mask is not None:
            assert attention_mask.ndim == 3, 'Mask dimension incorrect: {}'.format(
                attention_mask.shape)
            assert attention_mask.shape[-1] == encoder_output.shape[1], \
                'Attention mask and encoder output shape mismatch ({}, {})'.format(attention_mask.shape[-1],
                                                                                   encoder_output.shape[1])
            assert attention_mask.shape[-2] == target_embedding.shape[1], \
                'Attention mask and target embedding shape mismatch ({}, {})'.format(attention_mask.shape[-2],
                                                                                     target_embedding.shape[1])

        batch_size = tf.shape(target_embedding)[0]
        timesteps = target_embedding.shape.as_list()[1]
        future_mask = self.get_future_mask(batch_size, timesteps) \
            if (mask_future and not fast_decode) else None

        _, output = self.decoding_stack((encoder_output, target_embedding),
                                        self_attention_mask=future_mask,
                                        attention_mask=attention_mask,
                                        fast_decode=fast_decode)
        return output

    def get_future_mask(self, batch_size, sequence_length):
        """Mask future targets and padding

            :param batch_size: a TF Dimension
            :param sequence_length: a TF Dimension
            :param padding_mask: None or bool Tensor with shape [batch_size, sequence_length]

            :return mask: bool Tensor with shape [batch_size, sequence_length, sequence_length]
        """

        xind = tf.tile(tf.range(sequence_length)[
                       None, :], (sequence_length, 1))
        yind = tf.tile(tf.range(sequence_length)[
                       :, None], (1, sequence_length))
        mask = yind >= xind
        mask = tf.tile(mask[None], (batch_size, 1, 1))

        # if padding_mask is not None:
        #     mask = tf.logical_and(padding_mask[:, :, None], mask[None, :, :])
        #     mask = tf.logical_and(mask, padding_mask[:, None, :])

        return mask


class TransformerInputEmbedding(tf.keras.Model):

    def __init__(self,
                 embed_size: int,
                 discrete: bool,
                 n_symbols: Optional[int] = None,
                 dropout: Optional[float] = None,
                 batch_norm: bool = False,
                 embedding_initializer=None) -> None:
        super().__init__()
        if discrete:
            assert n_symbols is not None, 'n_symbols not passed in but model set to discrete'
            if embedding_initializer is not None:
                assert embedding_initializer.shape[0] == n_symbols, 'n_symbols and initializer shape mismatch'
                assert embedding_initializer.shape[1] == embed_size, 'embed_size, initializer shape mismatch'
                self.embedding = tf.keras.layers.Embedding(n_symbols, embed_size,
                                                           weights=[
                                                               embedding_initializer],
                                                           mask_zero=True)
            else:
                self.embedding = tf.keras.layers.Embedding(
                    n_symbols, embed_size)
        else:
            assert n_symbols is None, 'n_symbols passed in but model set to continuous'
            assert embedding_initializer is None, 'embedding_initializer passed in but model set to continouous'
            self.embedding = tf.keras.layers.Dense(
                embed_size, activation='relu')
        self.position_embedding = PositionEmbedding()
        self.dropout = None if dropout is None else tf.keras.layers.Dropout(
            dropout)
        self.batch_norm = None if batch_norm is False else tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        embedding = self.embedding(inputs)
        if self.dropout:
            embedding = self.dropout(embedding)
        if self.batch_norm:
            embedding = self.batch_norm(embedding)
        embedding = self.position_embedding(embedding)
        return embedding


class Transformer(tf.keras.Model):

    def __init__(self,
                 discrete: bool,
                 n_symbols_in: Optional[int] = None,
                 n_symbols_out: Optional[int] = None,
                 out_size: Optional[int] = None,
                 output_activation: Optional[str] = None,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_filter: int = 2048,
                 dropout: Optional[float] = None,
                 embedding_initializer=None) -> None:
        super().__init__()
        self.discrete = discrete
        if discrete:
            self.n_symbols_in = n_symbols_in
            self.n_symbols_out = n_symbols_out
        else:
            self.out_size = out_size
        self.output_activation = output_activation
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.dropout = dropout

        if not self.discrete:
            assert n_symbols_in is None, 'n_symbols_in passed in but model set to continuous'
            assert n_symbols_out is None, 'n_symbols_out passed in but model set to continuous'
            assert out_size is not None and out_size > 0, 'out_size not passed in but model set to continuous'
            assert embedding_initializer is None, 'embedding_initializer passed in but model set to continuous'
        else:
            assert n_symbols_in is not None, 'n_symbols_in not passed in but model set to discrete'
            assert n_symbols_out is not None, 'n_symbols_out not passed in but model set to discrete'
            assert out_size is None, 'out_size passed in but model set to discrete'
            assert embedding_initializer is not None, 'embedding_initializer not passed in but model set to discrete'

        self.input_embedding = TransformerInputEmbedding(d_model, discrete, n_symbols_in, dropout,
                                                         embedding_initializer=embedding_initializer)
        self.target_embedding = TransformerInputEmbedding(d_model, discrete, n_symbols_out, dropout,
                                                          embedding_initializer=embedding_initializer)

        self.encoder = TransformerEncoder(
            n_layers, n_heads, d_model, d_filter, dropout)
        self.decoder = TransformerDecoder(
            n_layers, n_heads, d_model, d_filter, dropout)

        self.output_layer = tf.keras.layers.Dense(
            n_symbols_out if discrete else out_size, activation=output_activation)

    def fast_decode(self, inputs, max_seq_len, padding_mask=None):
        target_size = 1 if self.discrete else self.out_size
        target_dtype = tf.int32 if self.discrete else tf.float32
        source_sequence = inputs
        batch_size = tf.shape(source_sequence)[0]
        output_sequence = [
            tf.zeros((batch_size, 1, target_size), dtype=target_dtype)]
        encoder_output = self.encode_source_sequence(
            source_sequence, padding_mask)
        for _ in range(max_seq_len):
            target_sequence = tf.concat(output_sequence, axis=1)
            output = self.decode_target_sequence(encoder_output,
                                                 target_sequence,
                                                 attention_mask=padding_mask,
                                                 shift_target_sequence_right=False,
                                                 training=True)
            output_sequence.append(output[:, -1:])
        return tf.concat(output_sequence[1:], axis=1)

    def encode_source_sequence(self, source_sequence, attention_mask):
        embedding = self.input_embedding(source_sequence)
        return self.encoder(embedding, attention_mask=attention_mask)

    def decode_target_sequence(self,
                               encoder_output,
                               target_sequence,
                               attention_mask,
                               shift_target_sequence_right,
                               training):
        if shift_target_sequence_right:
            batch_size, target_size = tf.shape(target_sequence)[
                0], target_sequence.shape.as_list()[-1]
            first_zeros = tf.zeros((batch_size, 1, target_size))
            target_sequence = tf.concat(
                (first_zeros, target_sequence[:, :-1]), axis=1)

        target_embedding = self.target_embedding(target_sequence)
        decoder_output = self.decoder((encoder_output, target_embedding),
                                      attention_mask=attention_mask,
                                      mask_future=training,
                                      fast_decode=not training)
        output = self.output_layer(decoder_output)
        return output

    def call(self, inputs, attention_mask=None, shift_target_sequence_right=True, training=True):
        source_sequence, target_sequence = inputs

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                encoder_mask = self._convert_padding_mask_to_attention_mask(
                    source_sequence, attention_mask)
                decoder_mask = self._convert_padding_mask_to_attention_mask(
                    target_sequence, attention_mask)
            elif attention_mask.ndim == 1:
                encoder_mask = self._convert_seqlens_to_attention_mask(
                    source_sequence, attention_mask)
                decoder_mask = self._convert_seqlens_to_attention_mask(
                    target_sequence, attention_mask)

        encoder_output = self.encode_source_sequence(
            source_sequence, attention_mask=encoder_mask)
        decoder_output = self.decode_target_sequence(encoder_output,
                                                     target_sequence,
                                                     attention_mask=decoder_mask,
                                                     shift_target_sequence_right=shift_target_sequence_right,
                                                     training=training)
        return decoder_output

    def _convert_padding_mask_to_attention_mask(self, inputs, mask):
        assert mask.shape[0] == inputs.shape[0], 'Mask and input batch size must match'
        assert mask.ndim == 2, 'Can only convert dimension 2 masks to dimension 3 masks'

        seqlen = inputs.shape[1]
        mask = tf.tile(mask[:, None, :], (1, seqlen, 1))
        return mask

    def _convert_seqlens_to_attention_mask(self, inputs, seqlens):
        assert seqlens.shape[0] == inputs.shape[0], 'Seqlens and input batch size must match'
        assert seqlens.ndim == 1, 'Can only convert dimension 1 seqlens to dimension 3 masks'

        indices = tf.tile(tf.range(inputs.shape[1])[
                          None, :], (seqlens.shape[0], 1))
        mask = indices < seqlens[:, None]
        return mask
