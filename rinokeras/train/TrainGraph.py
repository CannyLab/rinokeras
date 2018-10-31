from typing import Union, Callable, Tuple, Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.contrib.distribute import DistributionStrategy, OneDeviceStrategy

from .TestGraph import TestGraph
from .train_utils import Inputs, Outputs, Losses, Gradients
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
                 distribution_strategy: DistributionStrategy = OneDeviceStrategy,
                 **kwargs) -> None:

        self.optimizer = optimizer
        self.grads_function = grads_function
        self.return_grad_summaries = return_grad_summaries
        super().__init__(model, loss_function, inputs, return_loss_summaries=return_loss_summaries,
                         distribution_strategy=distribution_strategy, **kwargs)

    @classmethod
    def from_experiment(cls, experiment: Experiment, inputs: Union[Inputs, tf.data.Dataset], **kwargs):
        return cls(
            lambda inputs: experiment.build_model(experiment.models, inputs), experiment.optimizer,
            experiment.loss_function, experiment.grads_function, inputs, **kwargs)

    def build(self, *args, **kwargs):
        K.set_learning_phase(1)
        self._global_step = tf.train.get_or_create_global_step()

        def distributed_grads_fn(inputs):
            outputs = self.model(inputs)
            grads, loss_packed = self.loss_function(inputs, outputs)
            loss, losses = self._unpack_losses(loss_packed)
            return grads, loss, losses

        central_device = self.distribution_strategy.parameter_devices[0]
        with self.distribution_strategy.scope():
            grads, loss, losses = self.distribution_strategy.call_for_each_tower(
                distributed_grads_fn, self.inputs)

            reduced_total = self.distribution_strategy.reduce(
                tf.VariableAggregation.MEAN, loss, central_device)
            to_reduce = [(loss, central_device) for loss in losses]
            reduced_losses = self.distribution_strategy.batch_reduce(
                tf.VariableAggregation.MEAN, to_reduce)

            update_op = self.optimizer._distributed_apply(
                self.distribution_strategy, grads, global_step=self._global_step)

            self.total_loss = self.distribution_strategy.unwrap(reduced_total)[0]
            self.losses = tuple(self.distribution_strategy.unwrap(loss)[0] for loss in reduced_losses)
            self.grads = grads  # TODO: fix this for gradient summaries
            self.update_op = update_op

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

    def update(self, inputs: Optional[Inputs] = None) -> Losses:
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
