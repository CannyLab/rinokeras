from typing import Union, Callable, Tuple

import tensorflow as tf
import tensorflow.keras.backend as K

from .TestGraph import TestGraph, Inputs, Outputs, Losses, Gradients


class TrainGraph(TestGraph):
    """Sets up placeholders so that you can call trainer.train or trainer.loss as if you're in eager mode.

        Args:
            *args: Placeholders for positional arguments to loss function
            **kwargs: Placeholders for keyword arguments to loss function
    """

    def __init__(self,
                 model: Callable[[Inputs], Outputs],
                 optimizer: tf.train.Optimizer,
                 loss_function: Callable[[Tuple[Inputs, Outputs]], Losses],
                 grads_function: Callable[[Tuple[Inputs, Outputs]], Tuple[Losses, Gradients]],
                 inputs: Union[Inputs, tf.data.Dataset],
                 return_loss_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 **kwargs) -> None:

        self.optimizer = optimizer
        self.grads_function = grads_function
        self.return_grad_summaries = return_grad_summaries
        super().__init__(model, loss_function, inputs, return_loss_summaries=return_loss_summaries, **kwargs)

    def build(self, *args, **kwargs):
        K.set_learning_phase(1)
        grads, loss_packed = self.grads_function(*args, **kwargs)
        loss, losses = self._unpack_losses(loss_packed)

        self._global_step = tf.train.get_or_create_global_step()
        self.outputs = self.model(self.inputs)
        grads, loss_packed = self.grads_function(self.inputs, self.outputs)
        loss, losses = self._unpack_losses
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
