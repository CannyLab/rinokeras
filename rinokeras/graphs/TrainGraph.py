from typing import Sequence, Union, Callable, Dict, Tuple

import tensorflow as tf
import tensorflow.keras.backend as K

from .TestGraph import TestGraph


class TrainGraph(TestGraph):
    """Sets up placeholders so that you can call trainer.train or trainer.loss as if you're in eager mode.

        Args:
            *args: Placeholders for positional arguments to loss function
            **kwargs: Placeholders for keyword arguments to loss function
    """

    def __init__(self,
                 optimizer: tf.train.Optimizer,
                 loss_function: Callable,
                 grads_function: Callable,
                 loss_args: Sequence,
                 loss_kwargs: Dict,
                 *args,
                 return_loss_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 **kwargs) -> None:
        self.optimizer = optimizer
        self.grads_function = grads_function
        self.return_grad_summaries = return_grad_summaries
        super().__init__(loss_function, loss_args, loss_kwargs, *args,
                         return_loss_summaries=return_loss_summaries, **kwargs)

    def build(self, *args, **kwargs):
        K.set_learning_phase(1)
        grads, loss_packed = self.grads_function(*args, **kwargs)
        loss, losses = self._unpack_losses(loss_packed)

        update_op = self.optimizer.apply_gradients(grads)
        self.total_loss = loss
        self.losses = losses
        self.grads = grads
        self.update_op = update_op
        self.args_in = args
        self.kwargs_in = kwargs
        self.handle = None

    def create_summaries(self):
        if self.return_grad_summaries:
            with tf.name_scope('gradients'):
                for grad, var in self.grads:
                    name = var.name.replace(':', '_')
                    tf.summary.histogram(name, grad)
        super().create_summaries()

    @classmethod
    def from_dataset(cls,  # type: ignore
                     optimizer: tf.train.Optimizer,
                     loss_function: Callable,
                     grads_function: Callable,
                     dataset: tf.data.Dataset,
                     *args,
                     **kwargs):
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
        batch = iterator.get_next()

        loss_args: tuple = ()
        loss_kwargs: dict = {}

        if isinstance(batch, dict):
            loss_kwargs = batch
        elif isinstance(batch, list) or isinstance(batch, tuple):
            loss_args = tuple(batch)
        else:
            loss_args = (batch,)

        new_class = cls(optimizer, loss_function, grads_function, loss_args, loss_kwargs, *args, **kwargs)
        new_class.handle = handle
        return new_class

    def update(self, *args, **kwargs) -> Union[float, Tuple]:
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
            _, loss, summaries = self._run_tensor([self.update_op, self.losses, self.summaries], *args, **kwargs)
            return loss, summaries
        else:
            _, loss = self._run_tensor([self.update_op, self.losses], *args, **kwargs)
            return loss
