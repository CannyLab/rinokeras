import tensorflow as tf
import numpy as np

class Trainer(object):

    def __init__(self, obs_shape, ac_shape, policy, discrete, optimizer='adam'):
        self._obs_shape = obs_shape if not np.isscalar(obs_shape) else (obs_shape,)
        self._ac_shape = ac_shape
        self._policy = policy
        self._discrete = discrete
        if not discrete and np.isscalar(ac_shape):
            self._ac_shape = (ac_shape,)
        self._num_param_updates = 0
        if optimizer == 'adam':
            self._optimizer = tf.train.AdamOptimizer()
        elif optimizer == 'rmsprop':
            self._optimizer = tf.train.RMSPropOptimizer()
        else:
            raise ValueError("Unrecognized optimizer. Received {}.".format(optimizer))
    def _batch_norm(self, array, mean, var):
        array = array - mean
        array = array / (tf.sqrt(var) + 1e-10)
        return array

    def _loss_function(self, obs, act, *args):
        return NotImplemented

    def _grads_function(self, obs, act, *args):
        with tf.GradientTape() as tape:
            loss = self._loss_function(obs, act, *args)
        
        if isinstance(loss, tuple):
            losses = loss[1:]
            total_loss = loss[0]
        else:
            losses = loss
            total_loss = loss

        return tape.gradient(total_loss, self._policy.variables), losses

    def _train_on_batch(self, obs, act, *args, learning_rate=1e-3):
        grads, loss = self._grads_function(obs, act, *args)
        self._optimizer._lr = learning_rate
        self._optimizer.apply_gradients(zip(grads, self._policy.variables))
        return loss

    def train(self, batch, learning_rate):
        return NotImplemented

    @property
    def num_param_updates(self):
        return self._num_param_updates
