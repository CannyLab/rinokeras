from typing import Union, Callable, Tuple, Optional, Sequence, List, Any

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.contrib.distribute import DistributionStrategy, OneDeviceStrategy

from .TestGraph import TestGraph
from .train_utils import Inputs, Outputs, Losses
from rinokeras.core.v1x.train import Experiment
import rinokeras.compat.distributed as distlib


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
                 learning_rate: Union[float, Callable[[int], float]] = 1e-3,
                 return_loss_summaries: bool = False,
                 return_variable_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 gradient_clip_type: str = 'none',
                 gradient_clip_bounds: Union[float, Tuple[float, float]] = 1.0,
                 distribution_strategy: DistributionStrategy = OneDeviceStrategy('/gpu:0'),
                 **kwargs) -> None:

        self.optimizer = optimizer
        self.return_grad_summaries = return_grad_summaries
        self._learning_rate_func = learning_rate

        self._clip_gradients = self._get_gradient_clip_function(
            gradient_clip_type, gradient_clip_bounds)

        super().__init__(
            model, build_model, loss_function, inputs,
            return_loss_summaries=return_loss_summaries,
            return_variable_summaries=return_variable_summaries,
            distribution_strategy=distribution_strategy,
            **kwargs)

    @classmethod
    def from_experiment(cls, experiment: Experiment, inputs: Union[Inputs, tf.data.Dataset]):
        return cls(
            experiment.model, experiment.build_model, experiment.optimizer,
            experiment.loss_function, inputs, experiment.learning_rate,
            return_loss_summaries=experiment.return_loss_summaries,
            return_variable_summaries=experiment.return_variable_summaries,
            return_grad_summaries=experiment.return_grad_summaries,
            gradient_clip_type=experiment.gradient_clipping,
            gradient_clip_bounds=experiment.gradient_clipping_bounds,
            distribution_strategy=experiment.distribution_strategy)

    def _distributed_fn(self):
        # self._distributed_global_step = tf.train.get_or_create_global_step()

        def loss_fn(inputs):
            outputs = self.build_model(inputs)
            loss_packed = self.loss_function(inputs, outputs)
            loss, losses = self._unpack_losses(loss_packed)
            loss += sum(self.model.losses)
            return outputs, loss, losses

        def grads_fn(inputs):
            if tf.executing_eagerly():
                with tf.GradientTape() as tape:
                    outputs, loss, losses = loss_fn(inputs)
                grads = tape.gradient(loss, self.model.variables)
                grads = zip(grads, self.model.variables)
            else:
                outputs, loss, losses = loss_fn(inputs)
                with tf.control_dependencies(self.model.updates):
                    grads = self.optimizer.compute_gradients(loss, self.model.variables)

            grads = self._clip_gradients(grads)
            return grads, outputs, loss, losses

        self._distributed_grads, self._distributed_outputs, self._distributed_total_loss, self._distributed_losses = \
            distlib.call_for_each_device(self.distribution_strategy, grads_fn, self.inputs)

        self.update_op = self.optimizer._distributed_apply(
            self.distribution_strategy, self._distributed_grads)
            # global_step=self._distributed_global_step)

    def _reduce_distributed_ops(self):
        super()._reduce_distributed_ops()
        central_device = self.distribution_strategy.parameter_devices[0]

        to_reduce = [(grad, central_device) for grad, _ in self._distributed_grads]
        reduced_grads = self.distribution_strategy.batch_reduce(
            tf.VariableAggregation.SUM, to_reduce)

        self.grads = [(grad, var) for grad, var in zip(reduced_grads, self.model.variables)]

    def _initialize_graph(self):
        self._global_step = tf.train.get_or_create_global_step()
        self._update_global_step = self._global_step.assign(self._global_step + 1)
        K.set_learning_phase(1)
        with tf.variable_scope(self._name):
            learning_rate = self._learning_rate_func if isinstance(self._learning_rate_func, float) \
                else self._learning_rate_func(0)
            self._learning_rate = tf.get_variable(
                'learning_rate', shape=(), dtype=tf.float32,
                initializer=tf.constant_initializer(learning_rate), trainable=False)
            if not tf.executing_eagerly():
                self._update_learning_rate_ph = tf.placeholder(
                    tf.float32, shape=(), name='learning_rate_placeholder')
                self._update_learning_rate_op = tf.assign(
                    self._learning_rate, self._update_learning_rate_ph)

            self.optimizer = self._get_optimizer(self.optimizer)

    def _finalize_graph(self):
        super()._finalize_graph()
        self._default_operation = 'update'

    def _create_summaries(self):
        super()._create_summaries()
        if self.return_grad_summaries:
            for grad, var in self.grads:
                name = var.name.replace(':', '_')
                tf.summary.histogram(name, grad, collections=[self.summary_collection])

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
            'adagrad': tf.train.AdagradOptimizer,
            'proximal-adagrad': tf.train.ProximalAdagradOptimizer,
            'ftrl': tf.train.FtrlOptimizer,
            'adamax': tf.contrib.opt.AdaMaxOptimizer
        }

        if optimizer in optimizers:
            return optimizers[optimizer](learning_rate=self._learning_rate)
        else:
            raise ValueError("Unrecognized optimizer. Received {}.".format(optimizer))

    def run(self,
            ops: Union[str, Sequence[tf.Tensor]],
            inputs: Optional[Inputs] = None,
            return_outputs: bool = False) -> Any:
        if ops == 'default':
            ops = self._default_operation
        if ops == 'loss':
            return self.loss(inputs, return_outputs=return_outputs)
        elif ops == 'update':
            return self.update(inputs, return_outputs=return_outputs)
        elif isinstance(ops, str):
            raise ValueError("Unrecognized op on graph: {}".format(ops))
        else:
            return self._run_tensor(ops, inputs)

    def update(self, inputs: Optional[Inputs] = None, return_outputs: bool = False) -> Losses:
        """Updates the model with placeholders in graph mode.

        Args:
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function

        Returns:
            loss (Union[float, Tuple]): Model loss on input batch

        Raises:
            RuntimeError: If not run inside a tf.Session context
        """
        ops = [self.update_op, self._update_global_step, self.losses]
        if not isinstance(self._learning_rate_func, float):
            self.learning_rate = self._learning_rate_func(self.global_step)
        if return_outputs:
            ops.append(self.outputs)
        if self.return_loss_summaries or self.return_grad_summaries:
            ops.append(self.summaries)
        _, _, *result = self._run_tensor(ops, inputs)
        self.update_progress_bar(result[0])
        if len(result) == 1:
            result = result[0]
        return result

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
            sess.run(
                self._update_learning_rate_op, feed_dict={self._update_learning_rate_ph: value})
