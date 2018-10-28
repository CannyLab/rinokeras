from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence, Callable, Union, Dict, Any, Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm


class AbstractGraph(ABC):

    _num_graphs: int = 0

    def __init__(self,
                 loss_function: Callable,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self._name = self.__class__.__name__.lower()
        if AbstractGraph._num_graphs > 0:
            self._name += '_{}'.format(AbstractGraph._num_graphs)
        AbstractGraph._num_graphs += 1

        self.loss_function = loss_function
        self.progress_bar = None

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
            losses = (losses,)

        return total_loss, losses

    def add_progress_bar(self, data_len: Optional[int] = None, epoch_num: Optional[int] = None):
        desc = None if epoch_num is None else 'Epoch {:>3}'.format(epoch_num)
        progress_bar = tqdm(total=data_len, desc=desc, leave=False,
                            dynamic_ncols=True, smoothing=0.1)
        progress_bar.__enter__()
        self.progress_bar = progress_bar

    def update_progress_bar(self, postfix=None):
        if self.progress_bar is not None:
            self.progress_bar.update()
            self.progress_bar.set_postfix(postfix)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.progress_bar.__exit__()
        self.progress_bar = None
        return exc_type is None or exc_type == tf.errors.OutOfRangeError

    @abstractmethod
    def run(self, ops: Union[str, Sequence[tf.Tensor]], *args, **kwargs) -> Any:
        raise NotImplementedError("run op not implemented.")

    @abstractmethod
    def update(self, *args, **kwargs) -> Union[float, Tuple]:
        raise NotImplementedError("update op not implemented.")

    @abstractmethod
    def loss(self, *args, **kwargs) -> Union[float, Tuple]:
        raise NotImplementedError("loss op not implemented")


class EagerGraph(AbstractGraph):

    def __init__(self,
                 optimizer: tf.train.Optimizer,
                 loss_function: Callable,
                 grads_function: Callable,
                 *args,
                 **kwargs) -> None:
        assert tf.executing_eagerly(), "Cannot use EagerGraph when not in tf eager mode."
        super(EagerGraph, self).__init__(loss_function, *args, **kwargs)
        self.optimizer = optimizer
        self.grads_function = grads_function

    def run(self, ops: str, *args, **kwargs) -> Union[tf.Tensor, Tuple]:  # type: ignore
        if ops == 'update':
            return self.update(*args, **kwargs)
        elif ops == 'loss':
            return self.loss(*args, **kwargs)
        else:
            raise ValueError("Unknown argument for ops: {}. \
                In eager mode, can only automatically run the update and loss ops.".format(ops))

    def update(self, *args, **kwargs) -> Union[tf.Tensor, Tuple]:
        """Updates the model in eager mode.

        Args:
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function

        Returns:
            loss (Union[float, tf.Tensor]): Model loss on input batch
        """
        K.set_learning_phase(1)
        grads, loss = self.grads_function(*args, **kwargs)
        self.optimizer.apply_gradients(grads)
        return loss

    def loss(self, *args, **kwargs) -> Union[tf.Tensor, Tuple]:
        """Gets the loss of the model in eager mode.

        Args:
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function

        Returns:
            loss (Union[tf.Tensor, Tuple]): Model loss on input batch
        """
        K.set_learning_phase(0)
        loss = self.loss_function(*args, **kwargs)
        return loss


class TestGraph(AbstractGraph):
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
        super().__init__(loss_function, *args, **kwargs)
        self.return_loss_summaries = return_loss_summaries
        self.build(*loss_args, **loss_kwargs)
        self.create_summaries()

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


class MultiGPUGraph(TrainGraph):

    def __init__(self,
                 optimizer: tf.train.Optimizer,
                 loss_function: Callable,
                 grads_function: Callable,
                 loss_args: Sequence,
                 loss_kwargs: Dict,
                 *args,
                 num_gpus: int = 1,
                 **kwargs) -> None:
        self.num_gpus = num_gpus
        super(MultiGPUGraph, self).__init__(optimizer, loss_function, grads_function, loss_args, loss_kwargs,
                                            *args, **kwargs)

    def build(self, *args, **kwargs):
        graphs = []
        loss = []
        losses = []
        grads = []

        self.args_in = args
        self.kwargs_in = kwargs

        with tf.device('/cpu:0'):
            loss_args = self._split_nested_tensors(args)
            loss_kwargs = self._split_nested_tensors(kwargs)
            for gpu in range(self.num_gpus):
                with tf.device('/gpu:{}'.format(gpu)):
                    graph = RunGraph(self.optimizer, self.loss_function, self.grads_function,
                                     loss_args[gpu], loss_kwargs[gpu])
                    graphs.append(graph)
                    loss.append(graph.total_loss)
                    losses.append(graph.losses)
                    grads.append(graph.grads)
            self.graphs = graphs

            self.total_loss = self._average_tensors(loss)
            if isinstance(self.graphs[0].losses, list) or isinstance(self.graphs[0].losses, tuple):
                self.losses = [self._average_tensors(loss) for loss in zip(*losses)]
            else:
                self.losses = self._average_tensors(losses)

            self.grads = self._average_gradients(grads)
            self.update_op = self.optimizer.apply_gradients(self.grads)
        self.handle = None

    def _split_nested_tensors(self, tensors: List) -> List:
        if isinstance(tensors, tuple) or isinstance(tensors, list):
            splits: List[List[Any]] = list([] for _ in range(self.num_gpus))
            for tensor in tensors:
                split_tensors = self._split_nested_tensors(tensor)
                for i, split_t in enumerate(split_tensors):
                    splits[i].append(split_t)
            splits = list(list(split) for split in splits)
        elif isinstance(tensors, dict):
            splits = list({} for _ in range(self.num_gpus))
            for kw, tensor in tensors.items():
                split_tensors = self._split_nested_tensors(tensor)
                for i, split_t in enumerate(split_tensors):
                    splits[i][kw] = split_t

        else:
            split_batch_size = tf.shape(tensors)[0] // self.num_gpus
            splits = []
            for i in range(self.num_gpus):
                start = split_batch_size * i
                end = split_batch_size * (i + 1)
                if (i + 1) != self.num_gpus:
                    splits.append(tensors[start:end])
                else:
                    splits.append(tensors[start:])
            # splits = tf.split(tensors, self.num_gpus, axis=0)
        return splits

    def _average_tensors(self, tensors: Sequence) -> Any:
        return tf.reduce_mean(tf.concat([tensor[None] for tensor in tensors], 0), 0)

    def _average_gradients(self, grads_and_vars: Sequence) -> Sequence:
        assert len(grads_and_vars) == self.num_gpus, 'Length of grads_and_vars does not match number of GPUs'
        if self.num_gpus == 1:
            return grads_and_vars[0]

        average_grads = []
        for grad_and_var in zip(*grads_and_vars):
            grad = self._average_tensors([g for g, _ in grad_and_var])
            average_grads.append((grad, grad_and_var[0][1]))
        return average_grads
