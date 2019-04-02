
from typing import Tuple, Optional, Sequence, Union
import tensorflow as tf

from packaging import version


Gradients = Sequence[Tuple[Optional[tf.Tensor], tf.Variable]]


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


def get_optimizer(optimizer, **kwargs):
    if isinstance(optimizer, tf.train.Optimizer):
        return optimizer
    if not isinstance(optimizer, str):
        raise TypeError("Unrecognized input for optimizer. Expected TF optimizer or string. \
                         Received {}.".format(type(optimizer)))

    # Optimizer set is a bit different in 1.13
    if version.parse("1.12.1") < version.parse(tf.__version__):
        optimizers = {
            'adam': tf.train.AdamOptimizer,
            'rmsprop': tf.train.RMSPropOptimizer,
            'sgd': tf.train.GradientDescentOptimizer,
            'momentum': tf.train.MomentumOptimizer,
            'adadelta': tf.train.AdadeltaOptimizer,
            'adagrad': tf.train.AdagradOptimizer,
            'proximal-adagrad': tf.train.ProximalAdagradOptimizer,
            'ftrl': tf.train.FtrlOptimizer,
        }
    else:
        optimizers = {
            'adam': tf.train.AdamOptimizer,
            'rmsprop': tf.train.RMSPropOptimizer,
            'sgd': tf.train.GradientDescentOptimizer,
            'momentum': tf.train.MomentumOptimizer,
            'adadelta': tf.train.AdadeltaOptimizer,
            'adagrad': tf.train.AdagradOptimizer,
            'adagradda': tf.train.AdagradDAOptimizer,
            'proximal-adagrad': tf.train.ProximalAdagradOptimizer,
            'proximal-gd': tf.train.ProximalGradientDescentOptimizer,
            'ftrl': tf.train.FtrlOptimizer,
            'adamax': tf.contrib.opt.AdaMaxOptimizer,
        }


    if optimizer not in optimizers:
        raise ValueError(
            "Unrecognized optimizer. Received {}.".format(optimizer))
    return optimizers[optimizer](**kwargs)

