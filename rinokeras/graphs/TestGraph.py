from typing import Sequence, Union, Any, Callable, Dict, Tuple

import tensorflow as tf
import tensorflow.keras.backend as K

from .RinokerasGraph import RinokerasGraph

Inputs = Union[tf.Tensor, Sequence[tf.Tensor], Dict[str, tf.Tensor]]
Outputs = Union[tf.Tensor, Sequence[tf.Tensor], Dict[str, tf.Tensor]]
Losses = Union[tf.Tensor, Sequence[tf.Tensor]]
Gradients = Sequence[Tuple[tf.Tensor, tf.Variable]]


class TestGraph(RinokerasGraph):
    """

    """

    def __init__(self,
                 model: Callable[[Inputs], Outputs],
                 loss_function: Callable[[Tuple[Inputs, Outputs]], Losses],
                 inputs: Union[Inputs, tf.data.Dataset],
                 return_loss_summaries: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.handle = None
        if isinstance(inputs, tf.data.Dataset):
            self.handle = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.data.Iterator.from_string_handle(
                self.handle, inputs.output_types, inputs.output_shapes)
            inputs = self.iterator.get_next()
        self.inputs = inputs
        self.model = model
        self.loss_function = loss_function
        self.return_loss_summaries = return_loss_summaries
        self.build()
        self.create_summaries()

    def build(self):
        K.set_learning_phase(0)
        self._global_step = tf.train.get_or_create_global_step()
        self.outputs = self.model(self.inputs)
        loss_packed = self.loss_function(self.inputs, self.outputs)
        loss, losses = self._unpack_losses(loss_packed)

        self.total_loss = loss
        self.losses = losses
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

    def _get_feed_dict(self, inputs: Union[bytes, Inputs]) -> Dict[tf.placeholder, Any]:
        if self.handle is not None:
            assert isinstance(inputs, bytes), 'Must pass in only string handle to dataset'
            feed_dict = {self.handle: inputs}
            return feed_dict

        feed_dict = {}
        self._map_to_placeholders(self.inputs, inputs, feed_dict)
        return feed_dict

    def _run_tensor(self, ops: Union[tf.Tensor, Sequence[tf.Tensor]], inputs: Union[bytes, Inputs]) -> Any:
        """Runs the network for a specific tensor

        Args:
            ops (Union[tf.Tensor, Sequence[tf.Tensor]]): op or sequence of ops to run
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function

        Returns:
            Result of running ops

        Raises:
            RuntimeError: If not run inside a tf.Session context
        """
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")

        feed_dict = self._get_feed_dict(inputs)

        results = sess.run(ops, feed_dict=feed_dict)
        return results

    def run(self, ops: Union[str, Sequence[tf.Tensor]], inputs: Union[bytes, Inputs]) -> Any:
        if ops == 'default':
            ops = self._default_operation

        if ops == 'update':
            return self.update(inputs)
        elif ops == 'loss':
            return self.loss(inputs)
        else:
            return self._run_tensor(ops, inputs)

    def update(self, inputs: Union[bytes, Inputs]) -> Losses:
        raise RuntimeError("Called update on a TestGraph. To train the model, you must use a TrainGraph.")

    def loss(self, inputs: Union[bytes, Inputs]) -> Losses:
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

    @property
    def global_step(self) -> int:
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")
        return tf.train.global_step(sess, self._global_step)
