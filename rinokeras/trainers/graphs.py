from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Callable, Union, Dict, Any

import tensorflow as tf


class AbstractGraph(ABC):

    _num_graphs: int = 0

    def __init__(self, 
                 optimizer: tf.train.Optimizer, 
                 loss_function: Callable,
                 grads_function: Callable) -> None:
        super().__init__()
        self._name = self.__class__.__name__.lower()
        if AbstractGraph._num_graphs > 0:
            self._name += '_{}'.format(AbstractGraph._num_graphs)
        AbstractGraph._num_graphs += 1

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.grads_function = grads_function

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

        return total_loss, losses

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
                 grads_function: Callable) -> None:
        assert tf.executing_eagerly(), "Cannot use EagerGraph when not in tf eager mode."
        super(EagerGraph, self).__init__(optimizer, loss_function, grads_function)

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
        loss = self.loss_function(*args, **kwargs)
        return loss


class RunGraph(AbstractGraph):
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
                 loss_kwargs: Dict) -> None:
        super(RunGraph, self).__init__(optimizer, loss_function, grads_function)
        self.build(*loss_args, **loss_kwargs)

    def build(self, *args, **kwargs):
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

    @classmethod
    def from_dataset(cls,
                     optimizer: tf.train.Optimizer, 
                     loss_function: Callable,
                     grads_function: Callable,
                     dataset: tf.data.Dataset):
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

        new_class = cls(optimizer, loss_function, grads_function, loss_args, loss_kwargs)
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
        """Runs the 
        
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
        """Updates the model with placeholders in graph mode.
        
        Args:
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function
        
        Returns:
            loss (Union[float, Tuple]): Model loss on input batch
        
        Raises:
            RuntimeError: If not run inside a tf.Session context
        """
        _, loss = self._run_tensor([self.update_op, self.losses], *args, **kwargs)
        return loss

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
        loss = self._run_tensor(self.losses, *args, **kwargs)
        return loss


class MultiGPUGraph(RunGraph):

    def __init__(self, 
                 optimizer: tf.train.Optimizer, 
                 loss_function: Callable,
                 grads_function: Callable,
                 loss_args: Sequence,
                 loss_kwargs: Dict,
                 num_gpus: int = 1) -> None:
        self.num_gpus = num_gpus
        super(MultiGPUGraph, self).__init__(optimizer, loss_function, grads_function, loss_args, loss_kwargs)

    def build(self, *args, **kwargs):
        graphs = []
        loss = []
        losses = []
        grads = []

        self.args_in = args
        self.kwargs_in = kwargs

        with tf.device('/cpu:0'):
            loss_args = [tf.split(arg, self.num_gpus, axis=0) for arg in args]
            loss_kwargs = {kw: tf.split(arg, self.num_gpus, axis=0) for kw, arg in kwargs.items()}
            for gpu in range(self.num_gpus):
                with tf.device('/gpu:{}'.format(gpu)):
                    args = [arg[gpu] for arg in loss_args]
                    kwargs = {kw: arg[gpu] for kw, arg in loss_kwargs.items()}

                    graph = RunGraph(self.optimizer, self.loss_function, self.grads_function, args, kwargs)
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
