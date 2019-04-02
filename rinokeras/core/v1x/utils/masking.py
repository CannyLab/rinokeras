import tensorflow as tf
from warnings import warn

def convert_sequence_mask_to_attention_mask(sequence, sequence_mask):
    """Given a padded input tensor of sequences and a boolean mask for each position
    in the sequence, returns a 3D boolean mask for use in attention.

    Args:
        sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length_1, ndim]
        sequence_mask (tf.Tensor[bool]): Tensor of shape [batch_size, sequence_length_2]

    Returns:
        tf.Tensor[bool]: Tensor of shape [batch_size, sequence_length_1, sequence_length_2]
    """
    batch_assert = tf.assert_equal(tf.shape(sequence_mask)[0], tf.shape(sequence)[0],
                                   message='batch size mismatch between input sequence and  \
                                            sequence_mask')
    rank_assert = tf.assert_equal(tf.rank(sequence_mask), 2,
                                  message='Can only convert 2D position mask to 3D attention mask')

    with tf.control_dependencies([batch_assert, rank_assert]):
        attention_mask = tf.tile(
            sequence_mask[:, None, :], (1, tf.shape(sequence)[1], 1))

        return attention_mask


def convert_padding_mask_to_attention_mask(sequence, padding_mask):
    """ DEPRECATED: use convert_sequence_mask_to_attention_mask instead
    """
    warn("convert_padding_mask_to_attention_mask is deprecated, \
          please use convert_sequence_mask_to_attention_mask intead", DeprecationWarning)
    return convert_sequence_mask_to_attention_mask(sequence, padding_mask)


def convert_sequence_length_to_sequence_mask(sequence, sequence_lengths):
    """Given a padded input tensor of sequences and a tensor of lengths, returns
    a boolean mask for each position in the sequence indicating whether or not
    that position is padding.

    Args:
        sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
        sequence_lengths (tf.Tensor[int]): Tensor of shape [batch_size]

    Returns:
        tf.Tensor[bool]: Tensor of shape [batch_size, sequence_length]
    """
    batch_assert = tf.assert_equal(tf.shape(sequence_lengths)[0], tf.shape(sequence)[0],
                                   message='batch size mismatch between input sequence and  \
                                            sequence_lengths')
    rank_assert = tf.assert_equal(tf.rank(sequence_lengths), 1,
                                  message='Can only convert 1D sequence_lengths to 2D mask')

    with tf.control_dependencies([batch_assert, rank_assert]):
        indices = tf.tile(tf.range(tf.shape(sequence)[1])[
                          None, :], (tf.shape(sequence_lengths)[0], 1))
        mask = indices < sequence_lengths[:, None]

        return mask


def convert_to_attention_mask(sequence, mask):
    """Automatically convert from None/1D/2D/3D mask to a boolean 3D attention mask.
    Note this does NOT allow for varying the input mask during training. We could replace
    the python if statements with tensorflow conditionals to allow this, but for the
    moment this is really a helper function and assumes that the type of mask
    passed in is fixed.

    Args:
        sequence (tf.Tensor): Tensor of shape [batch_size, sequence_length, ndim]
        mask: Optional[Tensor] of shape [batch_size]
                                     or [batch_size, sequence_length]
                                     or [batch_size, sequence_length, sequence_length]

    Returns:
        Optional[tf.Tensor[bool]]: Tensor of shape [batch_size, sequence_length, sequence_length]
    """

    if mask is None:
        return None

    if len(mask.shape) == 1:
        mask = convert_sequence_length_to_sequence_mask(
            sequence, mask)

    if len(mask.shape) == 2:
        mask = convert_sequence_mask_to_attention_mask(
            sequence, mask)

    if mask.dtype != tf.bool:
        mask = tf.cast(mask, tf.bool)

    return mask
