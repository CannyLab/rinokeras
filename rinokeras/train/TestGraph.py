from typing import Sequence, Union, Any, Callable, Dict, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.contrib.distribute import DistributionStrategy, OneDeviceStrategy

from rinokeras.train import Experiment
from .RinokerasGraph import RinokerasGraph
from .train_utils import Inputs, Outputs, Losses


class TestGraph(RinokerasGraph):
    """
    Constructs a keras model and sets up ops to automatically run the loss function, etc.

    Args:
        model (Callable[[Inputs], Outputs]): Function that builds the model. This can be as simple
            as a keras model that maps inputs to outputs, or can perform arbitrarily complex operations.
            Importantly, it does not actually need to be a keras model.
        loss_function (Callable[[Tuple[Inputs, Outputs]], Losses]): Function that maps inputs and
            outputs to losses
        return_loss_summaries (bool, default=False): Whether to return TF summaries.
    """

    def __init__(self,
                 model: Model,
                 build_model: Callable[[Inputs], Outputs],
                 loss_function: Callable[[Inputs, Outputs], Losses],
                 inputs: Union[Inputs, tf.data.Dataset],
                 return_loss_summaries: bool = False,
                 distribution_strategy: DistributionStrategy = OneDeviceStrategy,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(inputs, tf.data.Dataset):
            inputs = distribution_strategy.distribute_dataset(lambda: inputs)
            self.iterator = inputs.make_initializable_iterator()
            inputs = self.iterator.get_next()
        self.distribution_strategy = distribution_strategy
        self.inputs = inputs
        self.model = model
        self.loss_function = loss_function
        self.return_loss_summaries = return_loss_summaries
        self.build()
        self.create_summaries()

    @classmethod
    def from_experiment(cls, experiment: Experiment, inputs: Union[Inputs, tf.data.Dataset]):
        return cls(
            experiment.model, experiment.build_model, experiment.loss_function, inputs,
            return_loss_summaries=experiment.return_loss_summaries,
            distribution_strategy=experiment.distribution_strategy)

    def build(self):
        K.set_learning_phase(0)

        def distributed_loss_fn(inputs):
            outputs = self.model(inputs)
            loss_packed = self.loss_function(inputs, outputs)
            loss, losses = self._unpack_losses(loss_packed)
            return loss, losses

        central_device = self.distribution_strategy.parameter_devices[0]
        with self.distribution_strategy.scope():
            loss, losses = self.distribution_strategy.call_for_each_tower(
                distributed_loss_fn, self.inputs)
            reduced_total = self.distribution_strategy.reduce(
                tf.VariableAggregation.MEAN, loss, central_device)
            to_reduce = [(loss, central_device) for loss in losses]
            reduced_losses = self.distribution_strategy.batch_reduce(
                tf.VariableAggregation.MEAN, to_reduce)
            self.total_loss = self.distribution_strategy.unwrap(reduced_total)[0]
            self.losses = tuple(self.distribution_strategy.unwrap(loss)[0] for loss in reduced_losses)

        self._default_operation = 'loss'

    def create_summaries(self):
        if self.return_loss_summaries:
            with tf.name_scope('losses'):
                for i, loss in enumerate(self.losses):
                    tf.summary.scalar(str(i), loss)
        self.summaries = tf.summary.merge_all()

    def _map_to_placeholders(self, placeholders, inputs, feed_dict):
        if isinstance(placeholders, tf.placeholder):
            feed_dict[placeholders] = inputs
        elif isinstance(placeholders, list) and isinstance(inputs, list):
            for ph, input_ in zip(placeholders, inputs):
                self._map_to_placeholders(ph, input_, feed_dict)
        elif isinstance(placeholders, tuple) and isinstance(inputs, tuple):
            for ph, input_ in zip(placeholders, inputs):
                self._map_to_placeholders(ph, input_, feed_dict)
        elif isinstance(placeholders, dict) and isinstance(inputs, dict):
            for key, ph in placeholders:
                self._map_to_placeholders(ph, inputs[key], feed_dict)
        else:
            raise ValueError("Type of placeholders and inputs did not match. Received \
                              {} and {}.".format(type(placeholders), type(inputs)))

    def _unpack_losses(self, losses: Losses):
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
            losses = (losses,)

        return total_loss, losses

    def run(self, ops: Union[str, Sequence[tf.Tensor]], inputs: Optional[Inputs] = None) -> Any:
        if ops == 'default':
            ops = self._default_operation

        if ops == 'loss':
            return self.loss(inputs)
        elif isinstance(ops, str):
            raise ValueError("Unrecognized op on graph: {}".format(ops))
        else:
            return self._run_tensor(ops, inputs)

    def loss(self, inputs: Optional[Inputs] = None) -> Losses:
        """Gets loss of model with placeholders in graph mode.

        Args:
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function

        Returns:
            loss (Union[float, Tuple]): Model loss on input batch

        Raises:
            RuntimeError: If not run inside a tf.Session context
        """
        if self.return_loss_summaries:
            return self._run_tensor([self.losses, self.summaries], inputs)
        else:
            return self._run_tensor(self.losses, inputs)

