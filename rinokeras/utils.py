"""
Various utility functions that are commonly used in our models and during training.
"""
import collections
from copy import copy
from collections import defaultdict
from typing import Optional, Sequence, Tuple, Union, Dict
from timeit import default_timer as timer
from warnings import warn

import tensorflow as tf
import tensorflow.keras.backend as K

Gradients = Sequence[Tuple[Optional[tf.Tensor], tf.Variable]]


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
    convert_sequence_mask_to_attention_mask(sequence, padding_mask)


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


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)

        return self.initial_p + fraction * (self.final_p - self.initial_p)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """See Schedule.value"""

        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)

                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None

        return self._outside_value


def clip_gradients(grads: Gradients, clip_type: str, clip_bounds: Union[float, Tuple[float, float]]) -> Gradients:

    def apply_clipping(g):
        if clip_type in ['none', 'None']:
            pass
        elif clip_type == 'value':
            assert isinstance(clip_bounds, (tuple, list)) and len(clip_bounds) == 2, \
                'Expected list or tuple of length 2, received {}'.format(
                    clip_bounds)
            g = tf.clip_by_value(g, clip_bounds[0], clip_bounds[1])
        elif clip_type in ['norm', 'global_norm', 'average_norm']:
            assert isinstance(clip_bounds, (int, float)) and clip_bounds > 0, \
                'Expected positive float, received {}'.format(clip_bounds)
            g = tf.clip_by_norm(g, clip_bounds)
        else:
            raise ValueError(
                "Unrecognized gradient clipping method: {}.".format(clip_type))

        return g

    return [(apply_clipping(g), v) for g, v in grads if g is not None and v.trainable]


def get_optimizer(optimizer, learning_rate=1e-3):
    if isinstance(optimizer, tf.train.Optimizer):
        return optimizer
    elif not isinstance(optimizer, str):
        raise TypeError("Unrecognized input for optimizer. Expected TF optimizer or string. \
                         Received {}.".format(type(optimizer)))

    def momentum_opt(learning_rate):
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8)

    optimizers = {
        'adam': tf.train.AdamOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,
        'momentum': momentum_opt,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'proximal-adagrad': tf.train.ProximalAdagradOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'adamax': tf.contrib.opt.AdaMaxOptimizer,
    }

    if optimizer in optimizers:
        return optimizers[optimizer](learning_rate=learning_rate)
    else:
        raise ValueError(
            "Unrecognized optimizer. Received {}.".format(optimizer))


def load_distributed(distribution_strategy, model, filename, by_name=False):
    with distribution_strategy.scope():
        model.load_weights(filename, by_name=by_name)
        weights = model.get_weights()
        assign_ops = []

        for layer in model.layers:
            num_param = len(layer.weights)
            layer_weights = weights[:num_param]

            for sw, w in zip(layer.weights, layer_weights):
                assign_ops.append(distribution_strategy.unwrap(sw.assign(w)))
            weights = weights[num_param:]
        K.get_session().run(assign_ops)


def get_shape(array, dim):
    if isinstance(dim, collections.Iterable):
        return [tf.shape(array)[d] if array.shape[d].value is None else array.shape[d].value for d in dim]
    else:
        return tf.shape(array)[dim] if array.shape[dim].value is None else array.shape[dim].value


def gather_from_last(array, indices):
    rank = array.shape.ndims
    dims = get_shape(array, range(rank - 1))
    range_indices = [tf.range(dim) for dim in dims]
    tile = dims + [indices.shape[-1]]
    tiled_indices = []
    for currdim, ind in enumerate(range_indices):
        for dim in range(rank):
            if dim != currdim:
                ind = tf.expand_dims(ind, dim)
        currtile = copy(tile)
        currtile[currdim] = 1
        tiled_indices.append(tf.tile(ind, currtile))

    indices = tf.stack(tiled_indices + [indices], -1)
    return tf.gather_nd(array, indices)


def accuracy(labels, predictions, weights=None, dtype=tf.float32):
    is_correct = tf.equal(labels, predictions)
    is_correct = tf.cast(is_correct, dtype)
    if weights is None:
        return tf.reduce_mean(is_correct)
    else:
        weights = tf.ones_like(is_correct) * weights
        return tf.reduce_sum(is_correct * weights) / tf.reduce_sum(weights)


class MetricsAccumulator(object):

    def __init__(self):
        self._totalmetrics = defaultdict(lambda: 0.0)
        self._nupdates = 0
        self._start_time = float('nan')

    def add(self, metrics: Dict[str, float]):
        for metric, value in metrics.items():
            self._totalmetrics[metric] += value
        self._nupdates += 1

    def start_timer(self):
        self._start_time = timer()

    def end_timer(self):
        self.runtime = timer() - self._start_time

    def get_average(self):
        assert self.nupdates > 0
        return {metric: value / self.nupdates for metric, value in self._totalmetrics.items()}

    def __iter__(self):
        return iter(self.get_average())

    def items(self):
        return self.get_average().items()

    def __getitem__(self, value: str) -> float:
        if value not in self._totalmetrics:
            raise KeyError(value)
        return self._totalmetrics[value] / self.nupdates

    def __str__(self) -> str:
        return str(self.get_average())

    @property
    def nupdates(self) -> int:
        return self._nupdates
