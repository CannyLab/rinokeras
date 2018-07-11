import numpy as np
import tensorflow as tf

class Policy(tf.keras.Model):
    def __init__(self, obs_shape, ac_shape, discrete):
        super().__init__()
        self._obs_shape = obs_shape if not np.isscalar(obs_shape) else (obs_shape,)
        self._ac_shape = ac_shape
        self._discrete = discrete
        if discrete and not np.isscalar(ac_shape):
            raise ValueError("ac_shape must be scalar if mode is discrete")
        if np.isscalar(ac_shape):
            self._ac_shape = (ac_shape,)

    # Should return the action to perform when seeing obs
    def predict(self, obs):
        return NotImplemented

    # Used for policies with intra-episode memory
    def clear_memory(self):
        return

    @property
    def ac_shape(self):
        return self._ac_shape

    @property
    def obs_shape(self):
        return self._obs_shape
