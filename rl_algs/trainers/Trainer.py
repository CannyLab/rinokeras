import tensorflow as tf
import numpy as np

class Trainer(object):

    def __init__(self, obs_shape, ac_shape, policy, discrete):
        self._obs_shape = obs_shape if not np.isscalar(obs_shape) else (obs_shape,)
        self._ac_shape = ac_shape
        self._policy = policy
        self._discrete = discrete
        if not discrete and np.isscalar(ac_shape):
            self._ac_shape = (ac_shape,)
        self._scope = None
        self._num_param_updates = 0

    def _setup_placeholders(self):
        return NotImplemented

    def _setup_loss(self):
        return NotImplemented

    def _get_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("No default session found. Run this within a tf.Session context.")
        return sess

    def train(self, batch, learning_rate):
        return NotImplemented

    @property
    def scope(self):
        return self._scope

    @property
    def num_param_updates(self):
        return self._num_param_updates
