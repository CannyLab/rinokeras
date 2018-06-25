import tensorflow as tf
import numpy as np

class Trainer(object):

    def __init__(self, model, discrete, optimizer='adam'):
        self._model = model
        self._discrete = discrete

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

    def loss_function(self, features, labels, *args):
        raise NotImplementedError("Must implement a loss function.")

    def grads_function(self, features, labels, *args):
        with tf.GradientTape() as tape:
            loss = self.loss_function(features, labels, *args)
        
        if isinstance(loss, tuple):
            losses = loss[1:]
            total_loss = loss[0]
        else:
            losses = loss
            total_loss = loss

        return tape.gradient(total_loss, self._model.variables), losses

    def _train_on_batch(self, features, labels, *args, learning_rate=1e-3):
        grads, loss = self.grads_function(features, labels, *args)
        self._optimizer._lr = learning_rate
        self._optimizer.apply_gradients(zip(grads, self._model.variables))
        return loss

    def train(self, batch, learning_rate):
        raise NotImplementedError("Need to write a method to process a batch (to pass appropriate arguments to the loss function)")

    @property
    def num_param_updates(self):
        return self._num_param_updates
