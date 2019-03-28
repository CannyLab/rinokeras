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


if version.parse(tf.__version__) < version.parse("1.13"):
    call_for_each_device = _call_for_each_device_v112
    reduce = _reduce_v112
    ReduceOp = _reduce_op_112()
else:
    call_for_each_device = _call_for_each_device_v113
    reduce = _reduce_v113
    ReduceOp = _reduce_op_113()
