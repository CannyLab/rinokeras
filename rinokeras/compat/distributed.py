"""
Compatability utils for the distributed API
"""

import tensorflow as tf
from packaging import version

# Call for each device override
def _call_for_each_device_v112(strategy, *args, **kwargs):
    return strategy.call_for_each_tower(*args, **kwargs)
def _call_for_each_device_v113(strategy, *args, **kwargs):
    return strategy.call_for_each_replica(*args, **kwargs)
def _call_for_each_device_v114(strategy, fn, *args, **kwargs):
    return strategy.extended.call_for_each_replica(fn, args, **kwargs)

# Reduce operation
def _reduce_v112(strategy, *args, **kwargs):
    if 'destinations' not in kwargs.keys() is None:
        raise ValueError('Reduction requires destinations in Tensorflow <= 1.12')
    return strategy.reduce(*args, **kwargs)
def _reduce_v113(strategy, *args, **kwargs):
    if 'destinations' in kwargs.keys():
        kwargs.pop('destinations')
    return strategy.reduce(*args, **kwargs)

# Reduce Operations
class _reduce_op_112:
    @property
    def MEAN(self,):
        return tf.VariableAggregation.MEAN
    @property
    def SUM(self,):
        return tf.VariableAggregation.SUM

class _reduce_op_113:
    @property
    def MEAN(self,):
        return tf.distribute.ReduceOp.MEAN
    @property
    def SUM(self,):
        return tf.distribute.ReduceOp.SUM

def _num_devices_v112(strategy):
    return strategy.num_towers
def _num_devices_v113(strategy):
    return strategy.num_replicas_in_sync


def _distribute_dataset_v112_v113(strategy, inputs):
    return strategy.distribute_dataset(inputs)
def _distribute_dataset_v114(strategy, inputs):
    return strategy.experimental_distribute_dataset(inputs())

def _batch_reduce_v112_v113(strategy, reduce_op, to_reduce):
    return strategy.batch_reduce(reduce_op, to_reduce)
def _batch_reduce_v114(strategy, reduce_op, to_reduce):
    return strategy.extended.batch_reduce_to(reduce_op, to_reduce)


if version.parse(tf.__version__) < version.parse("1.13"):
    call_for_each_device = _call_for_each_device_v112
    reduce = _reduce_v112
    ReduceOp = _reduce_op_112()
    num_devices = _num_devices_v112
    distribute_dataset = _distribute_dataset_v112_v113
    batch_reduce = _batch_reduce_v112_v113
elif version.parse("1.13") <= version.parse(tf.__version__) < version.parse("1.14"):
    call_for_each_device = _call_for_each_device_v113
    reduce = _reduce_v113
    ReduceOp = _reduce_op_113()
    num_devices = _num_devices_v113
    distribute_dataset = _distribute_dataset_v112_v113
    batch_reduce = _batch_reduce_v112_v113
elif version.parse("1.14") <= version.parse(tf.__version__):
    call_for_each_device = _call_for_each_device_v114
    reduce = _reduce_v113
    ReduceOp = _reduce_op_113()
    num_devices = _num_devices_v113
    distribute_dataset = _distribute_dataset_v114
    batch_reduce = _batch_reduce_v114