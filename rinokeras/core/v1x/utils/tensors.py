
from copy import copy
import collections
import tensorflow as tf

def get_shape(array, dim):
    if isinstance(dim, collections.Iterable):
        return [tf.shape(array)[d] if array.shape[d].value is None else array.shape[d].value for d in dim]
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
