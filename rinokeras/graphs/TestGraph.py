from typing import Sequence, Union, Any, Callable, Dict, Tuple

import tensorflow as tf
import tensorflow.keras.backend as K

from .RinokerasGraph import RinokerasGraph


class TestGraph(RinokerasGraph):
    """Sets up placeholders so that you can call trainer.train or trainer.loss as if you're in eager mode.

        Args:
            *args: Placeholders for positional arguments to loss function
            **kwargs: Placeholders for keyword arguments to loss function
    """

    def __init__(self,
                 loss_function: Callable,
                 loss_args: Sequence,
                 loss_kwargs: Dict,
                 *args,
                 return_loss_summaries: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.return_loss_summaries = return_loss_summaries
        self.build(*loss_args, **loss_kwargs)
        self.create_summaries()
        self._default_operation = 'loss'

    def build(self, *args, **kwargs):
        K.set_learning_phase(0)
        loss_packed = self.loss_function(*args, **kwargs)
        loss, losses = self._unpack_losses(loss_packed)

        self.total_loss = loss
        self.losses = losses
        self.args_in = args
        self.kwargs_in = kwargs
        self.handle = None

    def create_summaries(self):
        if self.return_loss_summaries:
            with tf.name_scope('losses'):
                for i, loss in enumerate(self.losses):
                    tf.summary.scalar(str(i), loss)
        self.summaries = tf.summary.merge_all()

    @classmethod
    def from_dataset(cls,
                     loss_function: Callable,
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

        new_class = cls(loss_function, loss_args, loss_kwargs, *args, **kwargs)
        new_class.handle = handle
        return new_class

    def _get_feed_dict(self, *args, **kwargs) -> Dict[tf.placeholder, Any]:
        if self.handle is not None:
            assert len(kwargs) == 0, 'Graph is set up from a handle, not from placeholders.'
            assert len(args) == 1 and isinstance(args[0], bytes), 'Must pass in only string handle to dataset'
            feed_dict = {self.handle: args[0]}
            return feed_dict

        if len(args) != len(self.args_in):
            raise ValueError("Expected {} positional arguments, but received {}.".format(
                len(self.args_in), len(args))
            )

        if len(kwargs) != len(self.kwargs_in):
            raise ValueError("Expected {} keyword arguments, but received {}.".format(
                len(self.kwargs_in), len(kwargs))
            )

        feed_dict = {}
        for arg_in, arg in zip(self.args_in, args):
            feed_dict[arg_in] = arg
        for kw in self.kwargs_in:
            try:
                feed_dict[self.kwargs_in[kw]] = kwargs[kw]
            except KeyError:
                raise KeyError("Expected keyword argument '{}'".format(kw))
        return feed_dict

    def _run_tensor(self, ops: Union[tf.Tensor, Sequence[tf.Tensor]], *args, **kwargs) -> Any:
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

        feed_dict = self._get_feed_dict(*args, **kwargs)

        results = sess.run(ops, feed_dict=feed_dict)
        return results

    def run(self, ops: Union[str, Sequence[tf.Tensor]], *args, **kwargs) -> Any:
        if ops == 'default':
            ops = self._default_operation

        if ops == 'update':
            return self.update(*args, **kwargs)
        elif ops == 'loss':
            return self.loss(*args, **kwargs)
        else:
            return self._run_tensor(ops, *args, **kwargs)

    def update(self, *args, **kwargs) -> Union[float, Tuple]:
        raise RuntimeError("Called update on a TestGraph. To train the model, you must use a TrainGraph.")

    def loss(self, *args, **kwargs) -> Union[float, Tuple]:
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
            return self._run_tensor([self.losses, self.summaries], *args, **kwargs)
        else:
            return self._run_tensor(self.losses, *args, **kwargs)

    @property
    def global_step(self) -> int:
        global_step = tf.train.get_or_create_global_step()
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")
        return tf.train.global_step(sess, global_step)
