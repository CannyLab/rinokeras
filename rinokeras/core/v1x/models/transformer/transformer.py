
import tensorflow as tf
import rinokeras
import rinokeras.core.v1x as rk

from tensorflow.keras import Model
from tensorflow.python.keras import layers as layer_module
import numpy as np

from typing import Optional

from rinokeras.core.v1x.common.layers import PositionEmbedding, EmbeddingTranspose
from rinokeras.core.v1x.common.layers import WeightNormDense as Dense

from .transformer_embedding import TransformerInputEmbedding
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder

class Transformer(Model):

    def __init__(self,
                # Dummy
                _dummy = None,
                 
                 # Size
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_filter: int = 2048,

                 # Symbols/Output
                 discrete: bool = True,
                 n_symbols_in: Optional[int] = None,
                 n_symbols_out: Optional[int] = None,
                 out_size: Optional[int] = None,
                 output_activation: Optional[str] = None,
                 multiply_with_embedding_transpose=False,
                 output_layer=None,
                 
                 # Dropout
                 dropout: Optional[float] = None,
                 layer_dropout: Optional[float] = None,

                 # Embedding
                 embedding_initializer=None,
                 use_preembedded_vectors=False,
                 share_source_target_embedding=False,
                 concat_position_encoding=False,
                 position_encoding_expands_dims=True,
                 encoder_use_position_encoding=True,

                 # Utils
                 use_weight_norm=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if _dummy is not None:
            raise ValueError('Pass arguments to the transformer ONLY using keyword arguments')

        # Save sizes
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter

        # Not sure if we need to have the discrete/non-discrete versions
        # Working through this.
        self.discrete = discrete

        self.n_symbols_in = n_symbols_in
        self.n_symbols_out = n_symbols_out
        self.out_size = out_size
        self.output_activation = output_activation
        self.mtranspose = multiply_with_embedding_transpose
        self.output_layer = output_layer

        self.dropout_weight = 0 if dropout is None else dropout
        self.layer_dropout = layer_dropout


        self.embedding_initializer = embedding_initializer
        self.preembedded = use_preembedded_vectors
        self.share_source_target_embedding = share_source_target_embedding
        self.concat_position_encoding = concat_position_encoding
        self.position_encoding_expands_dims = position_encoding_expands_dims
        self.encoder_use_position_encoding = encoder_use_position_encoding
        
        self.use_weight_norm = use_weight_norm
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        # Handle the position encoding
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.extra_kwargs = kwargs

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
                activity_regularizer=self.activity_regularizer, concat_position_encoding=self.concat_position_encoding,
                reproject_position_encoding=not self.position_encoding_expands_dims, use_position_encoding=self.encoder_use_position_encoding)

            if not self.share_source_target_embedding:
                target_embedding = TransformerInputEmbedding(
                    d_model, discrete, n_symbols_out, dropout,
                    embedding_initializer=embedding_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    concat_position_encoding=self.concat_position_encoding,
                    reproject_position_encoding=not self.position_encoding_expands_dims)
            else:
                target_embedding = input_embedding
        else:
            input_embedding = PositionEmbedding(
                concat=self.concat_position_encoding,
                reproject_embedding=not self.position_encoding_expands_dims)

        # If the position encoding is concatenation, then we need to reshape
        # the overall model to handle the position-encoded elements

        if self.concat_position_encoding:
            if self.position_encoding_expands_dims:
                # There's two ways to handle this - one, we could add a dense layer too the elements,
                # or two, we could change the dimension of the model
                self.d_model *= 2  # This should handle the internal dimension shift

        if output_layer is None:
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
            n_layers, n_heads, self.d_model, d_filter, dropout, layer_dropout,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            use_weight_norm=use_weight_norm)

        # Build the decoder stack.
        self.decoder = TransformerDecoder(
            target_embedding, output_layer,
            n_layers, n_heads, self.d_model, d_filter, dropout, layer_dropout,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            use_weight_norm=use_weight_norm)

    # Test decoding. Does not use the fast-decode method (which would make this much more effective)
    def test_decode(self, source_sequence, max_seq_len, encoder_mask=None, initial_input=None, preembed_hook=None):
        if self.preembedded:
            if preembed_hook is None:
                raise ValueError(
                    'Need embedding hook for test-decode when using pre-embedded vectors')

        # TODO: Replace this with something more robust
        target_dtype = tf.int32 if self.discrete else tf.float32
        output_size = self.n_symbols_out if self.discrete else self.out_size

        # Generate the masks for the encoder and decoder. There are a lot of different ways that
        # the attention masks could be passed in, so this method handles a lot of these different
        # mask shapes.
        encoder_mask = rk.utils.convert_to_attention_mask(source_sequence, encoder_mask)
        # Compute the encoder output
        encoder_output = self.encoder(
            source_sequence, mask=encoder_mask)

        return self.decoder.fast_decode(encoder_output, max_seq_len, output_size=output_size,
                                        output_dtype=target_dtype, encoder_mask=encoder_mask,
                                        initial_input=initial_input, preembed_hook=preembed_hook)

    def beam_decode(self, source_sequence, max_seq_len, encoder_mask=None, initial_input=None, n_beams=4):
        # Compute the encoder output
        encoder_mask = rk.utils.convert_to_attention_mask(source_sequence, encoder_mask)
        encoder_output, _ = self.encoder(source_sequence, mask=encoder_mask)
        batch_size = tf.shape(encoder_output)[0]

        return self.decoder.fast_beam_decode(encoder_output, max_seq_len, batch_size, n_beams, initial_input=initial_input)

    def call(self, inputs, mask=None, shift_target_sequence_right=True, mask_future=True):

        # Unpack the source and target sequences from the encoder.
        # If we're discrete, then:
        # Source Sequence: [batch_size x source_length]
        # Target Sequence: [batch_size x target_length]
        #
        # If we're not discrete, then:
        # Source Sequence: [batch_size x source_length x input_feature_shape]
        # Target Sequence: [batch_size x target_length x output_feature_shape]
        source_sequence, target_sequence = inputs
        if mask is not None:
            encoder_mask, decoder_mask = mask

        # Generate the masks for the encoder and decoder. There are a lot of different ways that
        # the attention masks could be passed in, so this method handles a lot of these different
        # mask shapes.
        encoder_mask = rk.utils.convert_to_attention_mask(
            source_sequence, encoder_mask)
        decoder_mask = rk.utils.convert_to_attention_mask(
            target_sequence, decoder_mask)

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
        encoder_output = self.encoder(
            source_sequence, mask=encoder_mask)

        # Finally, we need to do a decoding this should generate a
        # tensor of shape [batch_size x target_length x d_model]
        # from the encoder output.
        decoder_output = self.decoder(
            inputs=(encoder_output, target_sequence),
            mask=(encoder_mask, decoder_mask),
            shift_target_sequence_right=shift_target_sequence_right, mask_future=mask_future,
            cache=None, seqpos=1)

        return decoder_output

    def get_config(self):
        ei = self.embedding_initializer.tolist() if self.embedding_initializer else None
        if self.output_layer is not None:
            olc = {
                'class_name': self.output_layer.__class__.__name__,
                'config': self.output_layer.get_config(),
            }
        else:
            olc = None

        config = {
            'discrete': self.discrete,
            'n_symbols_in': self.n_symbols_in,
            'n_symbols_out': self.n_symbols_out,
            'out_size': self.out_size,
            'output_activation': self.output_activation,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'd_filter': self.d_filter,
            'dropout': self.dropout_weight,
            'layer_dropout': self.layer_dropout,
            'embedding_initializer': ei,
            'use_preembedded_vectors': self.preembedded,
            'multiply_with_embedding_transpose': self.mtranspose,
            'share_source_target_embedding': self.share_source_target_embedding,
            'use_weight_norm': self.use_weight_norm,
            'concat_position_encoding': self.concat_position_encoding,
            'position_encoding_expands_dims': self.position_encoding_expands_dims,
            'encoder_use_position_encoding': self.encoder_use_position_encoding,
            'output_layer_config': olc,
            'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer),
        }
        return rinokeras.compat.utils.merge_dicts(config, self.extra_kwargs)

    @classmethod
    def from_config(cls, config):
        ei = config.pop('embedding_initializer')
        if ei:
            ei = np.array(ei)
        if config['output_layer_config'] is not None:
            output_layer = layer_module.deserialize(config.pop('output_layer_config'), custom_objects=globals())
        else:
            config.pop('output_layer_config')
            output_layer = None
        return cls(embedding_initializer=ei, output_layer=output_layer, **config)


