from typing import Union, Callable, Tuple

import tensorflow as tf
import tensorflow.keras.backend as K

from .TestGraph import TestGraph, Inputs, Outputs, Losses, Gradients
from rinokeras.train import Experiment


class TrainGraph(TestGraph):
    """
    Constructs a keras model and sets up ops to automatically run the loss function, etc.

    Args:
        model (Callable[[Inputs], Outputs]): Function that builds the model. This can be as simple
            as a keras model that maps inputs to outputs, or can perform arbitrarily complex operations.
            Importantly, it does not actually need to be a keras model.
        optimizer (tf.train.Optimizer): The TF optimizer to use
        loss_function (Callable[[Tuple[Inputs, Outputs]], Losses]): Function that maps inputs and
            outputs to losses
        grads_function (Callable[[Tuple[Inputs, Outputs]], Tuple[Gradients, Losses]]): Function that
            calls the loss function and returns gradients
        return_loss_summaries (bool, default=False): Whether to return TF summaries for losses.
        return_grad_summaries (bool, default=False): Whether to return TF summaries for gradients.
    """

    def __init__(self,
                 model: Callable[[Inputs], Outputs],
                 optimizer: tf.train.Optimizer,
                 loss_function: Callable[[Inputs, Outputs], Losses],
                 grads_function: Callable[[Inputs, Outputs], Tuple[Gradients, Losses]],
                 inputs: Union[Inputs, tf.data.Dataset],
                 return_loss_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 **kwargs) -> None:

        self.optimizer = optimizer
        self.grads_function = grads_function
        self.return_grad_summaries = return_grad_summaries
        super().__init__(model, loss_function, inputs, return_loss_summaries=return_loss_summaries, **kwargs)

    @classmethod
    def from_experiment(cls, experiment: Experiment, inputs: Union[Inputs, tf.data.Dataset], **kwargs):
        return cls(
            lambda inputs: experiment.build_model(experiment.models, inputs), experiment.optimizer,
            experiment.loss_function, experiment.grads_function, inputs, **kwargs)

    def build(self, *args, **kwargs):
        K.set_learning_phase(1)
        self._global_step = tf.train.get_or_create_global_step()
        self.outputs = self.model(self.inputs)
        grads, loss_packed = self.grads_function(self.inputs, self.outputs)
        loss, losses = self._unpack_losses(loss_packed)
        update_op = self.optimizer.apply_gradients(grads, global_step=self._global_step)

        self.total_loss = loss
        self.losses = losses
        self.grads = grads
        self.update_op = update_op
        self._default_operation = 'update'

    def create_summaries(self):
        if self.return_grad_summaries:
            with tf.name_scope('gradients'):
                for grad, var in self.grads:
                    name = var.name.replace(':', '_')
                    tf.summary.histogram(name, grad)
        super().create_summaries()

    def update(self, inputs: Union[bytes, Inputs]) -> Losses:
        """Updates the model with placeholders in graph mode.

        Args:
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function

        Returns:
            loss (Union[float, Tuple]): Model loss on input batch

        Raises:
            RuntimeError: If not run inside a tf.Session context
        """
        if self.return_loss_summaries or self.return_grad_summaries:
            _, loss, summaries = self._run_tensor([self.update_op, self.losses, self.summaries], inputs)
            return loss, summaries
        else:
            _, loss = self._run_tensor([self.update_op, self.losses], inputs)
            return loss
