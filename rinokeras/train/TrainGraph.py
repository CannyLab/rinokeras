from typing import Union, Callable, Tuple, Optional, Sequence, List, Any

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.contrib.distribute import DistributionStrategy, OneDeviceStrategy

from .TestGraph import TestGraph
from .train_utils import Inputs, Outputs, Losses, Gradients
from rinokeras.train import Experiment
from rinokeras.common import optimizers as rinokeras_optimizers


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
                 model: Model,
                 build_model: Callable[[Inputs], Outputs],
                 optimizer: tf.train.Optimizer,
                 loss_function: Callable[[Inputs, Outputs], Losses],
                 inputs: Union[Inputs, tf.data.Dataset],
                 learning_rate: float = 1e-3,
                 return_loss_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 gradient_clip_type: str = 'none',
                 gradient_clip_bounds: Union[float, Tuple[float, float]] = 1.0,
                 distribution_strategy: DistributionStrategy = OneDeviceStrategy,
                 **kwargs) -> None:

        self.optimizer = optimizer
        self.return_grad_summaries = return_grad_summaries

        with tf.variable_scope(self._name):
            self._learning_rate = tf.get_variable('learning_rate', shape=(), dtype=tf.float32,
                                                  initializer=tf.constant_initializer(learning_rate),
                                                  trainable=False)
            if not tf.executing_eagerly():
                self._update_learning_rate_ph = tf.placeholder(tf.float32, shape=(), name='learning_rate_placeholder')
                self._update_learning_rate_op = tf.assign(self._learning_rate, self._update_learning_rate_ph)

            self.optimizer = self._get_optimizer(optimizer)

        self._clip_gradients = self._get_gradient_clip_function(gradient_clip_type, gradient_clip_bounds)

        # Setup gradient clipping
        super().__init__(model, build_model, loss_function, inputs, return_loss_summaries=return_loss_summaries,
                         distribution_strategy=distribution_strategy, **kwargs)

    @classmethod
    def from_experiment(cls, experiment: Experiment, inputs: Union[Inputs, tf.data.Dataset]):
        return cls(
            experiment.model, experiment.build_model, experiment.optimizer,
            experiment.loss_function, inputs, experiment.learning_rate,
            return_loss_summaries=experiment.return_loss_summaries,
            return_grad_summaries=experiment.return_grad_summaries,
            gradient_clip_type=experiment.gradient_clipping,
            gradient_clip_bounds=experiment.gradient_clipping_bounds,
            distribution_strategy=experiment.distribution_strategy)

    def build(self, *args, **kwargs):
        K.set_learning_phase(1)

        def distributed_grads_fn(inputs):
            outputs = self.model(inputs)
            grads, loss_packed = self.grads_function(inputs, outputs)
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

    def grads_function(self,
                       inputs: Inputs,
                       outputs: Outputs,
                       model: Model) -> Tuple[Gradients, Losses]:
        """Computes the gradient of the loss function wrt model parameters.

        Args:
            inputs (Inputs): Inputs to the build_model function
            outputs (Outputs): Outputs from the build_model function

        Returns:
            Gradients: Gradients of the model parameters
            Losses: Different losses returned by loss_function
        """
        # TODO: fix this for multiple models
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                loss_packed = self.loss_function(inputs, outputs)
                total_loss, _ = self._unpack_losses(loss_packed)
                loss_to_optimize = total_loss + sum(model.losses)

            grads = tape.gradient(loss_to_optimize, model.variables)
            grads = zip(grads, model.variables)
        else:
            loss_packed = self.loss_function(inputs, outputs)
            total_loss, _ = self._unpack_losses(loss_packed)
            loss_to_optimize = total_loss + sum(model.losses)
            grads = self.optimizer.compute_gradients(loss_to_optimize, model.variables)

        # By default all of these norms use L2 TODO: Add additional norm types to the options
        grads = self._clip_gradients(grads)

        return grads, loss_packed

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

    def create_summaries(self):
        if self.return_grad_summaries:
            with tf.name_scope('gradients'):
                for grad, var in self.grads:
                    name = var.name.replace(':', '_')
                    tf.summary.histogram(name, grad)
        super().create_summaries()

    def run(self, ops: Union[str, Sequence[tf.Tensor]], inputs: Optional[Inputs] = None) -> Any:
        if ops == 'default':
            ops = self._default_operation

        if ops == 'loss':
            return self.loss(inputs)
        elif ops == 'update':
            return self.update(inputs)
        elif isinstance(ops, str):
            raise ValueError("Unrecognized op on graph: {}".format(ops))
        else:
            return self._run_tensor(ops, inputs)

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
