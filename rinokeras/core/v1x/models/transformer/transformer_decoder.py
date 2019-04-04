
import tensorflow as tf
import warnings


from collections import namedtuple
from typing import Optional

from tensorflow.keras import Model
from tensorflow.python.keras import layers as layer_module

from rinokeras.core.v1x.common.layers import Stack, LayerDropout

from .transformer_attention import TransformerMultiAttention
from .transformer_embedding import TransformerInputEmbedding
from .transformer_ff import TransformerFeedForward
from .transformer_utils import check_mask_shapes, get_cross_attention_mask, get_self_attention_mask
from .transformer_utils import shift_target_sequence_right as fn_shift_target_sequence_right


DecoderResult = namedtuple('DecoderResult', [
                           'seqpos', 'inputs', 'cache', 'output_sequence', 'is_finished'])
DecRes = namedtuple('DecRes', ['seqpos', 'inputs', 'cache', 'output_sequence', 'is_finished', 'seq_length', 'scores'])

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
                 layer_dropout: Optional[float] = None,
                 use_weight_norm=True,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer:  Optional[tf.keras.regularizers.Regularizer] = None,
                 ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.use_weight_norm = use_weight_norm

        # Get the initializers for serialization
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.self_attention = TransformerMultiAttention(
            n_heads, dropout,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        self.layer_drop_1 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)
        self.multi_attention = TransformerMultiAttention(
            n_heads, dropout,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        self.layer_drop_2 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)
        self.feed_forward = TransformerFeedForward(filter_size, hidden_size, 
                                                   dropout=dropout,
                                                   kernel_initializer=kernel_initializer,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                   use_weight_norm=use_weight_norm)
        self.layer_drop_3 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)

    def call(self, inputs, mask=None, **kwargs):

        # Unpack the inputs from the Keras API call
        encoder_outputs, decoder_inputs = inputs
        if mask is not None:
            self_attention_mask, cross_attention_mask = mask
        else:
            self_attention_mask = None
            cross_attention_mask = None

        return_self_attention_weights = kwargs.get('return_self_attention_weights', False)
        return_cross_attention_weights = kwargs.get('return_cross_attention_weights', False)

        # Unpack the decoder inputs
        if isinstance(decoder_inputs, tuple):
            decoder_inputs, cache = decoder_inputs
            start_cdn = tf.equal(cache['seqpos'], tf.constant(0))
            all_inputs = tf.cond(start_cdn, lambda: decoder_inputs, lambda: tf.concat([cache[self.name], decoder_inputs], axis=1))
            cache[self.name] = all_inputs
        else:
            all_inputs = decoder_inputs
            cache = None
        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        # Compute the selt-attention over the decoder inputs. This uses the self-attention
        # mask to control for the future outputs.
        # This generates a tensor of size [batch_size x target_len x d_model]
        target_selfattn, self_attention_weights = self.self_attention(
            (all_inputs, decoder_inputs),
            mask=self_attention_mask,
            return_attention_weights=True)
        target_selfattn = self.layer_drop_1(target_selfattn, decoder_inputs)

        # Compute the attention using the keys/values from the encoder, and the query from the
        # decoder. This takes the encoder output of size [batch_size x source_len x d_model] and the
        # target self-attention layer of size [batch_size x target_len x d_model] and then computes
        # a multi-headed attention across them, giving an output of [batch_size x target_len x d_model]
        # using the encoder as the keys and values and the target as the queries
        if encoder_outputs is not None:
            encdec_attention, cross_attention_weights = self.multi_attention(
                (encoder_outputs, target_selfattn),
                mask=cross_attention_mask,
                return_attention_weights=True)
            attn_output = self.layer_drop_2(encdec_attention, target_selfattn)
        else:
            attn_output = target_selfattn

        output = self.feed_forward(attn_output)
        output = self.layer_drop_3(output, attn_output)

        output = output if cache is None else (output, cache)

        if return_self_attention_weights and not return_cross_attention_weights:
            return output, self_attention_weights
        elif not return_self_attention_weights and return_cross_attention_weights:
            return output, cross_attention_weights
        elif return_self_attention_weights and return_cross_attention_weights:
            return output, self_attention_weights, cross_attention_weights
        else:
            return (encoder_outputs, output)

    def get_config(self):
        config = {
            'n_heads': self.n_heads,
            'filter_size': self.filter_size,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
            'layer_dropout': self.layer_dropout,
            'use_weight_norm': self.use_weight_norm,
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

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformerDecoder(Model):
    """Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    # TODO: Not sure about beam search, other methods of decoding for NLP.
    def __init__(self,
                 embedding_layer: Optional[Model],
                 output_layer: Optional[Model],
                 n_layers: int,
                 n_heads: int,
                 d_model: int,
                 d_filter: int,
                 dropout: Optional[float] = None,
                 layer_dropout: Optional[float] = None,
                 use_weight_norm:bool = True,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer:  Optional[tf.keras.regularizers.Regularizer] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_filter = d_filter
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.use_weight_norm = use_weight_norm

        # Save the regularizers for get_config
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.embedding_layer = embedding_layer
        if not isinstance(self.embedding_layer, TransformerInputEmbedding):
            raise AssertionError('Must have TransformerInputEmbedding layer as embedding layer')
        

        self.decoding_stack = Stack([TransformerDecoderBlock(n_heads, d_filter, d_model, dropout, layer_dropout,
                                                             kernel_initializer=kernel_initializer,
                                                             kernel_regularizer=kernel_regularizer,
                                                             bias_regularizer=bias_regularizer,
                                                             activity_regularizer=activity_regularizer,
                                                             use_weight_norm=use_weight_norm)

                                     for _ in range(n_layers)],
                                    name='decoder_blocks')
        self.output_layer = output_layer

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def call(self, inputs, mask=None, mask_future=False, shift_target_sequence_right=False, seqpos=1, cache=None):
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

        # Unpack inputs and masking
        encoder_output, target_input  = inputs
        if mask is not None:
            encoder_mask, decoder_mask = mask
        else:
            encoder_mask = None
            decoder_mask = None

        if shift_target_sequence_right:
            target_input = fn_shift_target_sequence_right(target_input)

        if self.embedding_layer is not None:
            target_embedding = self.embedding_layer(target_input, start=seqpos)
        else:
            target_embedding = target_input

        print(target_embedding)

        if cache is not None and mask_future:
            warnings.warn("Future masking should be unnecessary when using caching and will probably cause an error. \
                           If you think it's necessary, feel free to suppress this warning.")

        # Check the input and target dimensions
        target_embedding.shape.assert_has_rank(3)
        if encoder_output is not None:
            encoder_output.shape.assert_has_rank(3)

        with tf.control_dependencies(check_mask_shapes(encoder_mask, decoder_mask)):
            # Build the future-mask if necessary. This is an upper-triangular mask
            # which is used to prevent the network from attending to later timesteps
            # in the target embedding
            batch_size = tf.shape(target_embedding)[0]
            sequence_length = tf.shape(target_embedding)[1]
            self_attention_mask = get_self_attention_mask(
                batch_size, sequence_length, decoder_mask, mask_future)
            # Build the cross-attention mask. This is an upper-left block matrix which takes care of the masking
            # of the output shapes
            if encoder_output is not None:
                cross_attention_mask = get_cross_attention_mask(encoder_output, target_input, encoder_mask, decoder_mask)
            else:
                cross_attention_mask = None

            # Now actually do the decoding which should take us to the right dimension
            _, decoder_output = self.decoding_stack(
                (encoder_output, target_embedding if cache is None else (
                    target_embedding, cache)),
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask)

            if cache is not None:
                decoder_output, _ = decoder_output

            if self.output_layer is not None:
                output = self.output_layer(decoder_output)
            else:
                output = decoder_output

            return output

    def fast_decode(self, encoder_output, max_seq_len, output_size=None,
                    output_dtype=tf.float32, encoder_mask=None, initial_input=None,
                    preembed_hook=None, stopping_criterion=None):
        output_sequence = tf.TensorArray(output_dtype, size=max_seq_len)
        discrete = output_dtype in [tf.int32, tf.int64]
        batch_size = tf.shape(encoder_output)[0]

        if initial_input is None:
            if discrete:
                shape = (batch_size, 1)
            else:
                if output_size is None:
                    raise ValueError('Output size cannot be None if not discrete.')
                shape = (batch_size, 1) if discrete else (
                    batch_size, 1, output_size)
            initial_input = tf.zeros((shape), dtype=output_dtype)
        elif isinstance(initial_input, int):
            shape = (batch_size, 1) if discrete else (
                batch_size, 1, output_size)
            initial_input = initial_input * tf.ones((shape), dtype=output_dtype)

        if stopping_criterion is not None:
            assert callable(stopping_criterion), \
                'stopping_criterion must be a function that takes in the output at a timestep and returns \
                 whether the timestep has finished'

        def decoding_step(seqpos, inputs, cache, output_sequence, is_finished):
            if preembed_hook is not None:
                inputs = preembed_hook(inputs)

            output = self(inputs=(encoder_output, inputs),
                          mask=(encoder_mask, None),
                          shift_target_sequence_right=False,
                          mask_future=False,
                          cache=cache,
                          seqpos=seqpos + 1)

            cache['seqpos'] = seqpos + 1

            if discrete:
                output = tf.argmax(output, axis=-1, output_type=output_dtype)
            target_input = output
            output = tf.squeeze(output, 1)

            if stopping_criterion is not None:
                is_finished_new = stopping_criterion(output)
                assert is_finished_new.dtype == tf.bool, 'stopping_criterion must return a boolean tensor'
                is_finished = is_finished | is_finished_new

            result = DecoderResult(
                seqpos=seqpos + 1,
                inputs=target_input,
                cache=cache,
                output_sequence=output_sequence.write(seqpos, output),
                is_finished=is_finished)

            return result

        output_shape = (None, None) if discrete else (None, None, output_size)
        initial_cache, cache_shapes = self.get_initial_cache(batch_size)

        inputs = DecoderResult(
            seqpos=tf.constant(0),
            inputs=initial_input,
            cache=initial_cache,
            output_sequence=output_sequence,
            is_finished=tf.zeros((batch_size,), dtype=tf.bool))

        shapes = DecoderResult(
            seqpos=inputs.seqpos.shape,
            inputs=tf.TensorShape(output_shape),
            cache=cache_shapes,
            output_sequence=tf.TensorShape(None),
            is_finished=inputs.is_finished.shape)

        result = tf.while_loop(
            lambda seqpos, inputs, cache, output_sequence, is_finished: ~tf.reduce_all(
                is_finished, 0),
            decoding_step,
            inputs,
            shapes,
            maximum_iterations=max_seq_len
        )

        stack_shape = (1, 0) if discrete else (1, 0, 2)
        output = tf.transpose(result.output_sequence.stack(), stack_shape)

        return output

    def tile_for_beams(self, tensor, n_beams):
        shape = tf.shape(tensor)
        tensor =  tf.expand_dims(tensor, axis=1)
        tensor = tf.tile(tensor, [1,n_beams,1,1])
        tensor = tf.reshape(tensor, [shape[0]*n_beams, shape[1], tensor.shape[-1]])
        return tensor

    def fast_beam_decode(self, encoder_output, max_seq_len, batch_size, n_beams, output_dtype=tf.int32, initial_input=None, 
                            preembed_hook=None, stopping_criterion=None, encoder_mask=None, sample=False):
        
        if preembed_hook is not None:
            raise NotImplementedError("Prembedding hook is not supported in fast_beam_decode")

        def decoding_step(seqpos, inputs, cache, output_sequence, is_finished, seq_length, scores):
            start_cdn = tf.equal(cache['seqpos'], tf.constant(0))

            output = self(inputs=(encoder_output, inputs),
                          mask=(encoder_mask, None),
                          shift_target_sequence_right=False,
                          mask_future=False,
                          cache=cache,
                          seqpos=seqpos + 1)

            last_output_logits = output[:, -1, :]

            logit_shapes = last_output_logits.get_shape()
            vocab_size = logit_shapes[1]

            last_output_logits_logs = tf.nn.log_softmax(last_output_logits)

            if not sample:
                best_logits_2, best_indices_2 = tf.nn.top_k(last_output_logits_logs, k=n_beams, sorted=True, name=None)
            else:
                best_indices_2 = tf.cast(tf.multinomial(last_output_logits, num_samples=n_beams), dtype=tf.int32)
                
                flat_logits_logs = tf.reshape(last_output_logits_logs, [-1]) # Flatten the last_output_logits
                to_add_to_indeces = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size*vocab_size*n_beams, delta=vocab_size), [-1,1]), [1,n_beams]), [-1])
                flat_indices = to_add_to_indeces + tf.reshape(best_indices_2, [-1])
                gathered_scores = tf.gather(flat_logits_logs, flat_indices)
                best_logits_2 = tf.reshape(gathered_scores, [n_beams*batch_size, n_beams])
                print("here")

            # When flattened, this should include first n_beams words from beam 1, then n_beams words from beam 2, etc.
            flattened_best_indices = tf.reshape(best_indices_2, (-1,1))
            flattened_best_logits = tf.reshape(best_logits_2, (-1,1))

            modified_scores = scores

            expanded_finished = tf.reshape(tf.tile(tf.reshape(is_finished, [-1,1]),[1,n_beams]), [-1])
            expanded_original_scores = tf.reshape(tf.tile(tf.reshape(scores, [-1,1]),[1,n_beams]), [-1])
            expanded_modified_scores = tf.reshape(tf.tile(tf.reshape(modified_scores, [-1,1]),[1,n_beams]), [-1])

            score_delta = tf.squeeze(flattened_best_logits, 1) * (1-tf.cast(expanded_finished, tf.float32))
            expanded_original_scores += score_delta
            expanded_modified_scores += score_delta

            def start_fn_choose_beams():
                # Special case, we force to select the first k words for each beam. To allow tie-breaking
                return tf.range(n_beams*batch_size)

            def normal_fn_choose_beams():
                # We have to get the top k for each beam. Good luck
                folded_scores = tf.reshape(expanded_modified_scores, [batch_size, -1])
                chosen_beam_mscores, chosen_beam_ids = tf.nn.top_k(folded_scores, k=n_beams, sorted=True)
                beam_added = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size*n_beams*n_beams, delta=n_beams*n_beams), [-1,1]), [1,n_beams]), [-1])
                chosen_beam_indices = beam_added + tf.reshape(chosen_beam_ids, [-1])
                return chosen_beam_indices

            chosen_beam_indices = tf.cond(start_cdn, start_fn_choose_beams, normal_fn_choose_beams) 
            chosen_beam_scores = tf.gather(expanded_original_scores, chosen_beam_indices)

            chosen_from_beam_index = tf.cast(tf.math.floor(chosen_beam_indices / n_beams), dtype=tf.int32) # We need this for caching purposes

            # Rewrite the  output
            last_words_chosen = tf.gather(flattened_best_indices, chosen_beam_indices)
            shuffled_is_finished = tf.gather(is_finished, chosen_from_beam_index)
            last_words_chosen = last_words_chosen * tf.reshape(1-tf.cast(is_finished, tf.int32), (n_beams*batch_size, 1))


            def copy_mech(): return initial_input[:, seqpos+1]
            def choose_mech(): return  last_words_chosen
            copy_cdn = tf.less(seqpos, tf.shape(initial_input)[-1]-1)
            last_words_chosen = tf.cond(copy_cdn, copy_mech, choose_mech)

            def start_output_function():
                return last_words_chosen

            def normal_output_function():
                output_seq = tf.gather(output_sequence, chosen_from_beam_index, axis=0)
                output_seq = tf.concat([output_seq, last_words_chosen], axis=1)
                return output_seq

            output_sequence = tf.cond(start_cdn, start_output_function, normal_output_function)

            scores = chosen_beam_scores

            # Decide which beams are finished or not
            if stopping_criterion is not None:
                is_finished_new = tf.reduce_any(stopping_criterion(output_sequence), axis=1)
                assert is_finished_new.dtype == tf.bool, 'stopping_criterion must return a boolean tensor'
                is_finished = is_finished | is_finished_new  # Is_finished comes from the previous time step

            seq_length += (1-tf.cast(is_finished, dtype=tf.int32))

            # Rewrite the cache
            for k in cache.keys():
                if k != 'seqpos': # This works for input sequences and cross_attn woot
                    cache[k] = tf.gather(cache[k], chosen_from_beam_index, axis=0)

            cache['seqpos'] = seqpos + 1
            result = DecRes(seqpos=seqpos + 1, inputs=last_words_chosen, cache=cache, output_sequence=output_sequence, is_finished=is_finished, seq_length=seq_length, scores=scores)

            return result

        if encoder_output is not None:
            encoder_output = self.tile_for_beams(encoder_output, n_beams)
            if encoder_mask is not None:
                encoder_mask = self.tile_for_beams(encoder_mask, n_beams)


        if initial_input is None:
            initial_input = tf.zeros((batch_size*n_beams, 1), dtype=output_dtype)
        else:
            initial_input = tf.tile(initial_input, [n_beams,1])

        initial_cache, cache_shapes = self.get_initial_cache(batch_size)

        inputs = DecRes(
            seqpos=tf.constant(0),
            inputs=initial_input,
            cache=initial_cache,
            output_sequence=tf.zeros((1,batch_size*n_beams), dtype=tf.int32),
            is_finished=tf.zeros((batch_size*n_beams,), dtype=tf.bool),
            seq_length=tf.zeros((batch_size*n_beams,), dtype=tf.int32),
            scores=tf.zeros((batch_size*n_beams,),dtype=tf.float32))

        shapes = DecRes(
            seqpos=inputs.seqpos.shape,
            inputs=tf.TensorShape((None, None)),
            cache=cache_shapes,
            output_sequence=tf.TensorShape([None,None]),
            is_finished=inputs.is_finished.shape,
            seq_length=inputs.seq_length.shape,
            scores=inputs.scores.shape)

        end_cond = lambda seqpos, inputs, cache, output_sequence, is_finished, seq_length, scores: ~tf.reduce_all(is_finished, 0)
        result  = tf.while_loop(end_cond, decoding_step, inputs, shapes, maximum_iterations=max_seq_len)
        output_words = tf.reshape(result.output_sequence, [batch_size, n_beams, -1])
        scores = tf.reshape(result.scores, [batch_size, -1])

        return output_words, scores

    def get_initial_cache(self, batch_size):
        initial_cache = {}
        initial_cache = {layer.name: tf.zeros((batch_size, 1, self.d_model), dtype=tf.float32) for layer in self.decoding_stack.layers} # [0]
        initial_cache_shapes = {layer.name: tf.TensorShape([None, None, self.d_model]) for layer in self.decoding_stack.layers} # [0]
        initial_cache['seqpos'] = tf.constant(0, dtype=tf.int32)
        initial_cache_shapes['seqpos'] = tf.TensorShape(None)
        return initial_cache, initial_cache_shapes



    def get_config(self):
        config = {
            'embedding_layer_config': {
                'class_name': self.embedding_layer.__class__.__name__,
                'config': self.embedding_layer.get_config(),
            },
            'output_layer_config': {
                'class_name': self.output_layer.__class__.__name__,
                'config': self.output_layer.get_config(),
            },
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'd_filter': self.d_filter,
            'dropout': self.dropout,
            'layer_dropout': self.layer_dropout,
            'use_weight_norm': self.use_weight_norm,
            'kernel_initializer':
            tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer)
        }
        return config

    @classmethod
    def from_config(cls, config):
        embedding_layer = layer_module.deserialize(config.pop('embedding_layer_config'), custom_objects=globals())
        output_layer = layer_module.deserialize(config.pop('output_layer_config'), custom_objects=globals())
        return cls(embedding_layer=embedding_layer, output_layer=output_layer, **config)
        
        
