from typing import Optional, List
import warnings

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, BatchNormalization, Lambda
import tensorflow.keras.backend as K
import numpy as np

import rinokeras as rk
from rinokeras.common.layers import Stack, DenseStack, LayerNorm, PositionEmbedding, EmbeddingTranspose
from rinokeras.common.attention import MultiHeadAttention, SelfAttention


class TransformerSelfAttention(Model):

    def __init__(self,
                 n_heads: int,
                 dropout: Optional[float],
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(TransformerSelfAttention, self).__init__()
        self.norm = LayerNorm()
        self.self_attention = SelfAttention(
            'scaled_dot', n_heads, dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer
        )

    def call(self, inputs, mask):
        norm_input = self.norm(inputs)
        attention = self.self_attention(norm_input, mask=mask)
        return attention + inputs


class TransformerMultiAttention(Model):
    def __init__(self,
                 n_heads: int,
                 dropout: Optional[float],
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(TransformerMultiAttention, self).__init__()
        self.multi_attention = MultiHeadAttention(
            'scaled_dot', n_heads, dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer
        )
        self.norm = LayerNorm()

    def call(self, inputs, mask):
        target, source = inputs
        norm_target = self.norm(target)
        attention = self.multi_attention((norm_target, source), mask=mask)
        return attention + target


class TransformerFeedForward(Model):
    def __init__(self, filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float],
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(TransformerFeedForward, self).__init__()
        self.norm = LayerNorm()
        self.feed_forward = DenseStack([filter_size, hidden_size], output_activation=None,
                                       kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer,
                                       activity_regularizer=activity_regularizer)
        self.dropout_weight = 0 if dropout is None else dropout
        self.dropout = Dropout(self.dropout_weight)

    def call(self, inputs):
        norm_input = self.norm(inputs)
        dense_out = self.feed_forward(norm_input)
        dense_out = self.dropout(dense_out, training=self.dropout_weight > 0)
        return dense_out + inputs


class TransformerEncoderBlock(Model):
    """An encoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: Tensor with shape [batch_size, sequence_length, channels]

    :return: output: Tensor with same shape as input
    """

    def __init__(self,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(TransformerEncoderBlock, self).__init__()
        self.self_attention = TransformerSelfAttention(
            n_heads, dropout, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer)
        self.feed_forward = TransformerFeedForward(filter_size, hidden_size, dropout,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer)

    def call(self, inputs, self_attention_mask=None):

        # Perform a multi-headed self-attention across the inputs.
        res_attn = self.self_attention(inputs, mask=self_attention_mask)
        output = self.feed_forward(res_attn)
        return output


class TransformerDecoderBlock(Model):
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
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        self.self_attention = TransformerMultiAttention(
            n_heads, dropout, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer)
        self.multi_attention = TransformerMultiAttention(
            n_heads, dropout, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer)
        self.feed_forward = TransformerFeedForward(filter_size, hidden_size, dropout,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer)

    def call(self, inputs, self_attention_mask=None, cross_attention_mask=None):
        encoder_outputs, decoder_inputs, cache = inputs  # Parse the encoder outputs from the input tensor

        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        # Compute the selt-attention over the decoder inputs. This uses the self-attention
        # mask to control for the future outputs.
        # This generates a tensor of size [batch_size x target_len x d_model]
        if cache is not None:
            seqpos = cache['seqpos']
            cache[self.name] = cache[self.name].write(seqpos, K.squeeze(decoder_inputs, 1))
            all_inputs = cache[self.name].stack()
            all_inputs = tf.transpose(all_inputs, (1, 0, 2))
        else:
            all_inputs = decoder_inputs
        print('decoder_inputs', decoder_inputs.shape)
        target_selfattn = self.self_attention((decoder_inputs, all_inputs), mask=self_attention_mask)
        print('target_selfattn', target_selfattn.shape)

        # Compute the attention using the keys/values from the encoder, and the query from the
        # decoder. This takes the encoder output of size [batch_size x source_len x d_model] and the
        # target self-attention layer of size [batch_size x target_len x d_model] and then computes
        # a multi-headed attention across them, giving an output of [batch_size x target_len x d_model]
        # using the encoder as the keys and values and the target as the queries
        encdec_attention = self.multi_attention((target_selfattn, encoder_outputs), mask=cross_attention_mask)
        print('encdec_attention', encdec_attention.shape)
        output = self.feed_forward(encdec_attention)
        print('output', output.shape)
        return [encoder_outputs, output, cache]


class TransformerEncoder(Model):
    """
    Stack of TransformerEncoderBlocks. Performs repeated self-attention.
    """

    def __init__(self,
                 embedding_layer: Model,
                 n_layers: int,
                 n_heads: int,
                 d_model: int,
                 d_filter: int,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        # The encoding stack is a stack of transformer encoder blocks
        self.encoding_stack = Stack([TransformerEncoderBlock(n_heads, d_filter, d_model, dropout,
                                                             kernel_regularizer=kernel_regularizer,
                                                             bias_regularizer=bias_regularizer,
                                                             activity_regularizer=activity_regularizer)
                                     for _ in range(n_layers)],
                                    name='encoder_stack')

    def call(self, inputs, encoder_mask=None):
        """
            Args:
                inputs: Either a float32 or in32 Tensor with shape [batch_size, sequence_length, ndim]
                encoder_mask: a boolean Tensor with shape [batch_size, sequence_length, sequence_length]
            Returns:
                output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        inputs = self.embedding_layer(inputs)
        inputs.shape.assert_has_rank(3)

        # We need to make sure that the input shapes are correct for the mask
        if encoder_mask is not None:
            # Check the dimension of the mask
            encoder_mask.shape.assert_has_rank(3)

            last_two_dims_equal = tf.assert_equal(tf.shape(encoder_mask)[-1], tf.shape(encoder_mask)[-2],
                                                  message='Last two mask dimensions must match')
            sequence_len_match = tf.assert_equal(tf.shape(encoder_mask)[-2], tf.shape(inputs)[1],
                                                 message='Sequence dimension between mask and inputs do not match')
            with tf.control_dependencies([last_two_dims_equal, sequence_len_match]):
                output = self.encoding_stack(inputs, self_attention_mask=encoder_mask)
        else:
            # Compute the output of the encoding stack
            output = self.encoding_stack(inputs, self_attention_mask=encoder_mask)
        return output


class TransformerDecoder(Model):
    """Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    # TODO: Not sure about beam search, other methods of decoding for NLP.
    def __init__(self,
                 embedding_layer: Model,
                 output_layer: Model,
                 n_layers: int,
                 n_heads: int,
                 d_model: int,
                 d_filter: int,
                 dropout: Optional[float] = None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        self.decoding_stack = Stack([TransformerDecoderBlock(n_heads, d_filter, d_model, dropout,
                                                             kernel_regularizer=kernel_regularizer,
                                                             bias_regularizer=bias_regularizer,
                                                             activity_regularizer=activity_regularizer)
                                     for _ in range(n_layers)],
                                    name='decoder_blocks')
        self.output_layer = output_layer

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def call(self, inputs, encoder_mask=None, decoder_mask=None, mask_future=False,
             shift_target_sequence_right=False, seqpos=1):
        """
            Args:
                inputs: a tuple of (encoder_output, target_embedding)
                    encoder_output: a float32 Tensor with shape [batch_size, sequence_length, d_model]
                    target_input: either a int32 or float32 Tensor with shape [batch_size, target_length, ndims]
                    cache: Used for fast decoding, a dictionary of tf.TensorArray. None during training.
                mask_future: a boolean for whether to mask future states in target self attention

            Returns:
                a tuple of (encoder_output, output)
                    output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        encoder_output, target_input, cache = inputs
        if shift_target_sequence_right:
            target_input = self.shift_target_sequence_right(target_input)
        print('shifted_target_input', target_input.shape)
        target_embedding = self.embedding_layer(target_input, start=seqpos)
        print('embedded', target_embedding.shape)
        if cache is not None and mask_future:
            warnings.warn("Future masking should be unnecessary when using caching and will probably cause an error. \
                           If you think it's necessary, feel free to suppress this warning.")

        # Check the input and target dimensions
        target_embedding.shape.assert_has_rank(3)
        encoder_output.shape.assert_has_rank(3)
        with tf.control_dependencies(self.check_mask_shapes(encoder_mask, decoder_mask)):
            # Build the future-mask if necessary. This is an upper-triangular mask
            # which is used to prevent the network from attending to later timesteps
            # in the target embedding
            batch_size = tf.shape(target_embedding)[0]
            sequence_length = tf.shape(target_embedding)[1]
            self_attention_mask = self.get_self_attention_mask(batch_size, sequence_length, decoder_mask, mask_future)
            # Build the cross-attention mask. This is an upper-left block matrix which takes care of the masking
            # of the output shapes
            cross_attention_mask = self.get_cross_attention_mask(inputs, encoder_mask, decoder_mask)

            # Now actually do the decoding which should take us to the right dimension
            print('target_embedding', target_embedding.shape)
            _, decoder_output, cache = self.decoding_stack(
                (encoder_output, target_embedding, cache),
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask)
            output = self.output_layer(decoder_output)

            return output

    def fast_decode(self, encoder_output, max_seq_len, output_size=None,
                    output_dtype=tf.float32, encoder_mask=None, preembed_hook=None):

        output_sequence = tf.TensorArray(output_dtype, size=max_seq_len)
        discrete = output_dtype in [tf.int32, tf.int64]
        batch_size = tf.shape(encoder_output)[0]
        shape = (batch_size, 1) if discrete else (batch_size, 1, output_size)
        initial_input = tf.zeros((shape), dtype=output_dtype)

        def decoding_step(i, target_input, cache, output_sequence):
            print('target_input', target_input.shape)
            output = self((encoder_output, target_input, cache), encoder_mask=encoder_mask,
                          decoder_mask=None, shift_target_sequence_right=False,
                          mask_future=False, seqpos=i + 1)
            cache['seqpos'] = i + 1

            target_input = output

            if discrete:
                output = tf.argmax(output, axis=-1, output_dtype=output_dtype)
                if preembed_hook is not None:
                    target_input = preembed_hook(output)

            return i + 1, target_input, cache, output_sequence.write(i, tf.squeeze(output, 1))

        inputs = [tf.constant(0), initial_input, self.get_initial_cache(max_seq_len), output_sequence]
        shapes = [inputs[0].shape, tf.TensorShape((None, None, initial_input.shape[-1])),
                  {name: getattr(el, 'shape', tf.TensorShape(None)) for name, el in inputs[2].items()}, tf.TensorShape(None)]
        _, _, _, output_sequence = tf.while_loop(
            lambda i, *_: i < max_seq_len,
            decoding_step,
            inputs,
            shapes
        )

        output = tf.transpose(output_sequence.stack(), (1, 0, 2))
        return output

    def shift_target_sequence_right(self, target_sequence: tf.Tensor) -> tf.Tensor:
        constant_values = 0 if target_sequence.dtype in [tf.int32, tf.int64] else 1e-10
        pad_array = [[0, 0] for _ in target_sequence.shape]
        pad_array[1][0] = 1
        target_sequence = tf.pad(target_sequence, pad_array, constant_values=constant_values)[:, :-1]
        return target_sequence

    def check_mask_shapes(self, encoder_mask, decoder_mask) -> List:
        # Make sure the decoder mask matches the correct embedding setup
        assertions = []
        if encoder_mask is not None:
            encoder_mask.shape.assert_has_rank(3)
            # Last two dimensions should match
            last_two_encoder_dims_equal = tf.assert_equal(tf.shape(encoder_mask)[-1], tf.shape(encoder_mask)[-2],
                                                          message='Last two encoder mask dimensions must match')
            assertions.append(last_two_encoder_dims_equal)
        if decoder_mask is not None:
            decoder_mask.shape.assert_has_rank(3)
            # Last two dimensions should match
            last_two_decoder_dims_equal = tf.assert_equal(tf.shape(decoder_mask)[-1], tf.shape(decoder_mask)[-2],
                                                          message='Last two encoder mask dimensions must match')
            assertions.append(last_two_decoder_dims_equal)
        return assertions

    def get_future_mask(self, batch_size, sequence_length):
        """Mask future targets and padding

            :param batch_size: a TF Dimension
            :param sequence_length: a TF Dimension
            :param padding_mask: None or bool Tensor with shape [batch_size, sequence_length]

            :return mask: bool Tensor with shape [batch_size, sequence_length, sequence_length]
        """

        xind = tf.tile(tf.range(sequence_length)[None, :], (sequence_length, 1))
        yind = tf.tile(tf.range(sequence_length)[:, None], (1, sequence_length))
        mask = yind >= xind
        mask = tf.tile(mask[None], (batch_size, 1, 1))

        return mask

    def get_self_attention_mask(self, batch_size, sequence_length, decoder_mask, mask_future):
        if not mask_future:
            return decoder_mask
        elif decoder_mask is None:
            return self.get_future_mask(batch_size, sequence_length)
        else:
            return decoder_mask & self.get_future_mask(batch_size, sequence_length)

    # This is an upper left block matrix which masks the attention for things that don't
    # exist within the internals.
    def get_cross_attention_mask(self, inputs, encoder_mask, decoder_mask):
        if encoder_mask is None and decoder_mask is None:
            cross_attention_mask = None
        elif encoder_mask is None:
            # We need to not mask the encoding, but mask the decoding
            # The decoding mask should have shape [batch_size x target_len x target_len]
            # meaning all we have to do is pad the mask out properly
            cross_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                                (1, tf.shape(inputs[0])[1], 1)), (0, 2, 1))
        elif decoder_mask is None:
            cross_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                                (1, 1, tf.shape(inputs[1])[1])), (0, 2, 1))
        else:
            dec_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                              (1, tf.shape(inputs[0])[1], 1)), (0, 2, 1))
            enc_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                              (1, 1, tf.shape(inputs[1])[1])), (0, 2, 1))
            cross_attention_mask = tf.logical_and(enc_attention_mask, dec_attention_mask)

        return cross_attention_mask

    def get_initial_cache(self, size):
        cache = {layer.name: tf.TensorArray(tf.float32, 1, dynamic_size=True, clear_after_read=False) for layer in
                 self.decoding_stack.layers[0]}
        cache['seqpos'] = tf.constant(0, dtype=tf.int32)
        return cache


# TODO: Split this into a discrete/continuous embedding rather than handle the logic here
class TransformerInputEmbedding(Model):

    def __init__(self,
                 embed_size: int,
                 discrete: bool,
                 n_symbols: Optional[int] = None,
                 dropout: Optional[float] = None,
                 batch_norm: bool = False,
                 n_embed_layers: int = 1,
                 embedding_initializer=None,
                 freeze_embeddings=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super(TransformerInputEmbedding, self).__init__()
        self.embedding_dense = Lambda(lambda x: x)
        self.using_dense_embedding = False
        if discrete:
            assert n_symbols is not None, 'n_symbols not passed in but model set to discrete'
            assert n_embed_layers == 1, 'discrete models can only have one embedding layer'
            if embedding_initializer is not None:
                assert embedding_initializer.shape[0] == n_symbols, \
                    'n_symbols and initializer shape mismatch'
                if embedding_initializer.shape[1] != embed_size:
                    # We have to correct if the input embedding isn't quite right
                    self.embedding = Embedding(n_symbols, embedding_initializer.shape[1],
                                               weights=[embedding_initializer],
                                               mask_zero=True,
                                               trainable=not freeze_embeddings)
                    self.embedding_dense = Dense(embed_size)
                    self.using_dense_embedding = True
                else:
                    self.embedding = Embedding(n_symbols, embed_size,
                                               weights=[embedding_initializer],
                                               mask_zero=True)
            else:
                self.embedding = Embedding(n_symbols, embed_size)
        else:
            assert n_symbols is None, 'n_symbols passed in but model set to continuous'
            assert embedding_initializer is None, 'embedding_initializer passed in but model set to continouous'
            self.embedding = DenseStack([embed_size] * n_embed_layers, output_activation='relu',
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer)

        self.discrete = discrete
        self.freeze_embeddings = freeze_embeddings
        self.position_encoding = PositionEmbedding()
        self.dropout_weight = 0 if dropout is None else dropout
        self.dropout = Dropout(self.dropout_weight)
        self.batch_norm = None if batch_norm is False else BatchNormalization()

    def call(self, inputs, start=1):

        # Compute the actual embedding of the inputs by using the embedding layer
        # TODO: Make sure that for non-discrete embeddings, this is handled correctly
        # and allow the shape to be correctly sorted. This should have a tensor
        # as output with shape [batch_size x sequence_len x d_model]
        embedding = self.embedding(inputs)
        if self.freeze_embeddings:
            embedding = K.stop_gradient(embedding)
        embedding = self.embedding_dense(embedding)
        embedding = self.dropout(embedding, self.dropout_weight > 0)

        if self.batch_norm:
            embedding = self.batch_norm(embedding, training=True)

        print(embedding.shape)
        embedding = self.position_encoding(embedding, start=start)
        print(embedding.shape)
        return embedding


class Transformer(Model):

    def __init__(self,
                 discrete: bool = True,
                 n_symbols_in: Optional[int] = None,
                 n_symbols_out: Optional[int] = None,
                 out_size: Optional[int] = None,
                 output_activation: Optional[str] = None,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_filter: int = 2048,
                 dropout: Optional[float] = None,
                 embedding_initializer=None,
                 use_preembedded_vectors=False,
                 multiply_wtih_embedding_transpose=False,
                 share_source_target_embedding=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        # Not sure if we need to have the discrete/non-discrete versions
        # Working through this.
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
        self.dropout_weight = 0 if dropout is None else dropout
        self.preembedded = use_preembedded_vectors
        self.mtranspose = multiply_wtih_embedding_transpose
        self.share_source_target_embedding = share_source_target_embedding

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        # Discrete model => Embedding Initializer/n-in/n-out
        # It's probably better to use a different word than 'discrete' to handle this
        # probably maybe kinda sorta

        if not self.discrete:
            assert n_symbols_in is None, 'n_symbols_in passed in but model set to continuous'
            assert n_symbols_out is None, 'n_symbols_out passed in but model set to continuous'
            assert out_size is not None and out_size > 0, 'out_size not passed in but model set to continuous'
            assert embedding_initializer is None, 'embedding_initializer passed in but model set to continuous'
        else:
            assert n_symbols_out is not None, 'n_symbols_out not passed in but model set to discrete'
            assert out_size is None, 'out_size passed in but model set to discrete'
            if not self.preembedded:
                assert n_symbols_in is not None, 'n_symbols_in not passed in but model set to discrete'
                assert embedding_initializer is not None, \
                    'embedding_initializer not passed in but model set to discrete'
                if self.share_source_target_embedding:
                    assert n_symbols_in == n_symbols_out, \
                        'n_symbols_in != n_symbols_out but share_source_target_embedding set'

        # Compute the input and target embedding layers.
        # This happens in both settings. If we're discrete, the embedding initializer
        # is None, while otherwise we initialize the embedding with the passed in weights.
        # In practice, we need to be able to freeze the embedding weights, which is a feature that we will have
        # to add soon.

        if not self.preembedded:
            input_embedding = TransformerInputEmbedding(
                d_model, discrete, n_symbols_in, dropout, embedding_initializer=embedding_initializer,
                kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer)
            if not self.share_source_target_embedding:
                target_embedding = TransformerInputEmbedding(
                    d_model, discrete, n_symbols_out, dropout, embedding_initializer=embedding_initializer,
                    kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
                    activity_regularizer=self.activity_regularizer)
            else:
                target_embedding = input_embedding
        else:
            input_embedding = PositionEmbedding()

        if self.mtranspose:
            output_layer = EmbeddingTranspose(target_embedding.embedding)
        else:
            output_layer = Dense(
                n_symbols_out if discrete else out_size, activation=output_activation,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer)

        # Build the encoder stack.
        self.encoder = TransformerEncoder(
            input_embedding,
            n_layers, n_heads, d_model, d_filter, dropout,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer)

        # Build the decoder stack.
        self.decoder = TransformerDecoder(
            target_embedding, output_layer,
            n_layers, n_heads, d_model, d_filter, dropout,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer)

    # Test decoding. Does not use the fast-decode method (which would make this much more effective)
    def test_decode(self, source_sequence, max_seq_len, encoder_mask=None, preembed_hook=None):
        if self.preembedded:
            if preembed_hook is None:
                raise ValueError('Need embedding hook for test-decode when using pre-embedded vectors')

        target_dtype = tf.int32 if self.discrete else tf.float32  # TODO: Replace this with something more robust

        # Generate the masks for the encoder and decoder. There are a lot of different ways that
        # the attention masks could be passed in, so this method handles a lot of these different
        # mask shapes.
        encoder_mask = rk.utils.convert_to_attention_mask(source_sequence, encoder_mask)
        print(encoder_mask.shape)
        # Compute the encoder output
        encoder_output = self.encoder(source_sequence, encoder_mask=encoder_mask)
        return self.decoder.fast_decode(encoder_output, max_seq_len, output_size=self.out_size,
                                        output_dtype=target_dtype, encoder_mask=encoder_mask,
                                        preembed_hook=preembed_hook)

    def call(self, inputs, encoder_mask=None, decoder_mask=None, shift_target_sequence_right=True, training=True):

        # Unpack the source and target sequences from the encoder.
        # If we're discrete, then:
        # Source Sequence: [batch_size x source_length]
        # Target Sequence: [batch_size x target_length]
        #
        # If we're not discrete, then:
        # Source Sequence: [batch_size x source_length x input_feature_shape]
        # Target Sequence: [batch_size x target_length x output_feature_shape]
        source_sequence, target_sequence = inputs
        mask_future = training

        # Generate the masks for the encoder and decoder. There are a lot of different ways that
        # the attention masks could be passed in, so this method handles a lot of these different
        # mask shapes.
        encoder_mask = rk.utils.convert_to_attention_mask(source_sequence, encoder_mask)
        decoder_mask = rk.utils.convert_to_attention_mask(target_sequence, decoder_mask)

        # After the end of the encoder and decoder generation phase, we have
        # Encoder Mask: [batch_size x source_length x source_length]
        # Decoder Mask: [batch_size x target_length x target_length]
        if self.preembedded:
            source_sequence.shape.assert_has_rank(3)
            target_sequence.shape.assert_has_rank(3)
            source_sequence.shape[2].assert_is_compatible_with(self.d_model)
            target_sequence.shape[2].assert_is_compatible_with(self.d_model)

        # Next, we perform the encoding of the sentence. This should take
        # as input a tensor of shape [batch_size x source_length x input_feature_shape]
        # and generate a tensor of shape [batch_size x source_length x d_model]
        encoder_output = self.encoder(source_sequence, encoder_mask=encoder_mask)

        # Finally, we need to do a decoding this should generate a
        # tensor of shape [batch_size x target_length x d_model]
        # from the encoder output.
        decoder_output = self.decoder(
            (encoder_output, target_sequence, None), encoder_mask=encoder_mask, decoder_mask=decoder_mask,
            shift_target_sequence_right=shift_target_sequence_right, mask_future=mask_future,
            seqpos=1)

        return decoder_output
