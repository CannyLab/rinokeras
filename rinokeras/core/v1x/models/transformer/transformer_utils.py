"""
Utilities for masking and shifting in transformers
"""

import tensorflow as tf
from typing import List

def shift_target_sequence_right(target_sequence: tf.Tensor) -> tf.Tensor:
    constant_values = 0 if target_sequence.dtype in [
        tf.int32, tf.int64] else 1e-10
    pad_array = [[0, 0] for _ in target_sequence.shape]
    pad_array[1][0] = 1
    target_sequence = tf.pad(
        target_sequence, pad_array, constant_values=constant_values)[:, :-1]

    return target_sequence

def check_mask_shapes(encoder_mask, decoder_mask) -> List:
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
                                                        message='Last two decoder mask dimensions must match')
        assertions.append(last_two_decoder_dims_equal)

    return assertions

def get_future_mask(batch_size, sequence_length):
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

    return mask

def get_self_attention_mask(batch_size, sequence_length, decoder_mask, mask_future):
    if not isinstance(mask_future, tf.Tensor) and not mask_future:
        return decoder_mask
    mask_future = mask_future if isinstance(mask_future, tf.Tensor) else \
        get_future_mask(batch_size, sequence_length)

    if decoder_mask is None:
        return mask_future

    return decoder_mask & mask_future

# This is an upper left block matrix which masks the attention for things that don't
# exist within the internals.
def get_cross_attention_mask(encoder_output, decoder_input, encoder_mask, decoder_mask):
    if encoder_mask is None and decoder_mask is None:
        cross_attention_mask = None
    elif encoder_mask is None:
        # We need to not mask the encoding, but mask the decoding
        # The decoding mask should have shape [batch_size x target_len x target_len]
        # meaning all we have to do is pad the mask out properly
        cross_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                                    (1, tf.shape(encoder_output)[1], 1)), (0, 2, 1))
    elif decoder_mask is None:
        cross_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                                    (1, 1, tf.shape(decoder_input)[1])), (0, 2, 1))
    else:
        dec_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                                    (1, tf.shape(encoder_output)[1], 1)), (0, 2, 1))
        enc_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                                    (1, 1, tf.shape(decoder_input)[1])), (0, 2, 1))
        cross_attention_mask = tf.logical_and(
            enc_attention_mask, dec_attention_mask)

    return cross_attention_mask
