from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Callable, Sequence, List

import tensorflow as tf
from tensorflow.keras import Model
import time
import contextlib
import os

from rinokeras.common import optimizers as rinokeras_optimizers
from .train_utils import Inputs, Outputs, Losses, Gradients


class Experiment(ABC):

    _num_experiments: int = 0

    def __init__(self,
                 models: Union[Model, Sequence[Model]],
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-3,
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Union[float, Tuple[float, ...]] = (-1, 1),
                 return_loss_summaries: bool = False,
                 return_grad_summaries: bool = False) -> None:
        super().__init__()
        self._name = self.__class__.__name__.lower()
        if Experiment._num_experiments > 0:
            self._name += '_{}'.format(Experiment._num_experiments)
        Experiment._num_experiments += 1

        if isinstance(models, Model):
            self.models = models
            self._models = [models]
        elif isinstance(models, tuple) or isinstance(models, list):
            self.models = list(models)
            self._models = list(models)
        else:
            raise TypeError(
                "Unrecognized input for models. Expected Model or list of Models, \
                 Received {}".format(type(models)))

        self.return_loss_summaries = return_loss_summaries
        self.return_grad_summaries = return_grad_summaries

        with tf.variable_scope(self._name):
            self._learning_rate = tf.get_variable('learning_rate', shape=(), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(learning_rate),
                                                  trainable=False)
            if not tf.executing_eagerly():
                self._update_learning_rate_ph = tf.placeholder(tf.float32, shape=(), name='learning_rate_placeholder')
                self._update_learning_rate_op = tf.assign(self._learning_rate, self._update_learning_rate_ph)

            self.optimizer = self._get_optimizer(optimizer)

        # Setup gradient clipping
        self._clip_gradients = self._get_gradient_clip_function(gradient_clipping, gradient_clipping_bounds)

    def _get_optimizer(self, optimizer):
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
            'proximal-adagrad': tf.train.ProximalAdagradOptimizer,
            'ftrl': tf.train.FtrlOptimizer,
            'adamax': rinokeras_optimizers.AdaMaxOptimizer,
        }

        if optimizer in optimizers:
            return optimizers[optimizer](learning_rate=self._learning_rate)
        else:
            raise ValueError("Unrecognized optimizer. Received {}.".format(optimizer))

    def _get_gradient_clip_function(self, clip_type: str, clip_bounds: Union[float, Tuple[float, ...]]) -> \
            Callable[[Sequence], List]:

        def clip_func(grads):
            clipped_grads = []
            for g, v in grads:
                if g is None:
                    # Choosing not to add gradients to list if they're None. Both adding/not adding are valid choices.
                    # clipped_grads.append((None, v))
                    continue
                if not v.trainable:
                    continue
                if clip_type in ['none', 'None']:
                    pass
                elif clip_type == 'value':
                    g = tf.clip_by_value(g, clip_bounds[0], clip_bounds[1])
                elif clip_type == 'norm':
                    g = tf.clip_by_norm(g, clip_bounds)
                elif clip_type == 'global_norm':
                    g = tf.clip_by_global_norm(g, clip_bounds)
                elif clip_type == 'average_norm':
                    g = tf.clip_by_average_norm(g, clip_bounds)
                else:
                    raise ValueError("Unrecognized gradient clipping method: {}.".format(clip_type))
                clipped_grads.append((g, v))
            return clipped_grads
        return clip_func

    @abstractmethod
    def build_model(self, model: Union[Model, Sequence[Model]], inputs: Inputs) -> Outputs:
        """Function that builds the model. This can be as simple as passing the inputs to the outputs
        or perform arbitrarily complex operations. Designed to be as flexible as you want.

        The function should return whatever
        outputs you need for computing the loss and any metrics you would like.

        Args:
            model (Union[Model, Sequence[Model]]): The input model here is the same as what you
                pass in to experiment on construction. This can be a single Keras model or a list of Models.

            inputs (Inputs): Tensor, List, or Dict of Tensors. Should generally be a placeholder or the
                             result of iterator.get_next() for some tf.data.Iterator

        Returns:
            Outputs: Tensor or Sequence of Tensors. First should be the total loss to optimize, followed
                     by any metrics you want to return.
        """
        raise NotImplementedError("Must implement a build_model function.")

    @abstractmethod
    def loss_function(self, inputs: Inputs, outputs: Outputs) -> Losses:
        """Loss function to be implemented by subclasses.

        Args:
            inputs (Inputs): Inputs to the build_model function
            outputs (Outputs): Outputs from the build_model function

        Returns:
            Losses: Different losses returned by loss_function

        Raises:
            NotImplementedError: Trainer is an abstract class, must subclass and implement loss_function
        """
        raise NotImplementedError("Must implement a loss function.")

    def grads_function(self, inputs: Inputs, outputs: Outputs) -> Tuple[Gradients, Losses]:
        """Computes the gradient of the loss function wrt model parameters.

        Args:
            inputs (Inputs): Inputs to the build_model function
            outputs (Outputs): Outputs from the build_model function

        Returns:
            Gradients: Gradients of the model parameters
            Losses: Different losses returned by loss_function
        """
        # TODO: fix this for multiple models
        variables = sum((model.variables for model in self._models), [])
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                loss_packed = self.loss_function(inputs, outputs)
                total_loss, _ = self._unpack_losses(loss_packed)
                loss_to_optimize = total_loss + sum(sum(model.losses) for model in self._models)

            grads = tape.gradient(loss_to_optimize, variables)
            grads = zip(grads, variables)
        else:
            loss_packed = self.loss_function(inputs, outputs)
            total_loss, _ = self._unpack_losses(loss_packed)
            loss_to_optimize = total_loss + sum(sum(model.losses) for model in self._models)
            grads = self.optimizer.compute_gradients(loss_to_optimize, variables)

        # By default all of these norms use L2 TODO: Add additional norm types to the options
        grads = self._clip_gradients(grads)

        return grads, loss_packed

    def _unpack_losses(self, losses: Union[tf.Tensor, Sequence[tf.Tensor]]):
        """Optionally unpacks a sequence of losses

        Args:
            losses (Union[tf.Tensor, Sequence[tf.Tensor]]): Loss tensor or sequence of loss tensors with
                first tensor being total loss

        Returns:
            tf.Tensor, Union[tf.Tensor, Sequence[tf.Tensor]]: Total loss, and sequence of loss tensors
        """
        if isinstance(losses, tuple) or isinstance(losses, list):
            total_loss = losses[0]
        else:
            total_loss = losses

        return total_loss, losses

    @property
    def learning_rate(self) -> tf.Variable:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if tf.executing_eagerly():
            self._learning_rate.assign(value)
        else:
            sess = tf.get_default_session()
            if sess is None:
                raise RuntimeError("Must be executed inside tf.Session context.")
            sess.run(self._update_learning_rate_op, feed_dict={self._update_learning_rate_ph: value})
