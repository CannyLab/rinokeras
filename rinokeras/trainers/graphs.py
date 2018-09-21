from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Callable, Union, Dict, Any

import tensorflow as tf


class AbstractGraph(ABC):

    _num_graphs: int = 0

    def __init__(self, 
                 optimizer: tf.train.Optimizer, 
                 loss_function: Callable,
                 grads_function: Callable,
                 learning_rate: float = 1e-3) -> None:
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
                 grads_function: Callable,
                 learning_rate: float = 1e-3) -> None:
        assert tf.executing_eagerly(), "Cannot use EagerGraph when not in tf eager mode."
        super(EagerGraph, self).__init__(optimizer, loss_function, grads_function, learning_rate)

    def run(self, ops: str, *args, **kwargs) -> Union[tf.EagerTensor, Tuple]:  # type: ignore
        if ops == 'update':
            return self.update(*args, **kwargs)
        elif ops == 'loss':
            return self.loss(*args, **kwargs)
        else:
            raise ValueError("Unknown argument for ops: {}. \
                In eager mode, can only automatically run the update and loss ops.".format(ops))

    def update(self, *args, **kwargs) -> Union[tf.EagerTensor, Tuple]:
        """Updates the model in eager mode.
        
        Args:
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function
        
        Returns:
            loss (Union[float, tf.EagerTensor]): Model loss on input batch
        """
        grads, loss = self.grads_function(*args, **kwargs)
        self.optimizer.apply_gradients(grads)
        return loss

    def loss(self, *args, **kwargs) -> Union[tf.EagerTensor, Tuple]:
        """Gets the loss of the model in eager mode.
        
        Args:
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function
        
        Returns:
            loss (Union[tf.EagerTensor, Tuple]): Model loss on input batch
        """
        loss = self.loss_function(*args, **kwargs)
        return loss


class PlaceholderGraph(AbstractGraph):

    def __init__(self, 
                 optimizer: tf.train.Optimizer, 
                 loss_function: Callable,
                 grads_function: Callable,
                 learning_rate: float = 1e-3) -> None:
        super(PlaceholderGraph, self).__init__(optimizer, loss_function, grads_function, learning_rate)
        self._built: bool = False

    def build(self, *args, **kwargs) -> None:
        """Sets up placeholders so that you can call trainer.train or trainer.loss as if you're in eager mode.
            
            Args:
                *args: Placeholders for positional arguments to loss function
                **kwargs: Placeholders for keyword arguments to loss function
        """
        grads, loss_packed = self.grads_function(*args, **kwargs)
        loss, losses = self._unpack_losses(loss_packed)

        update_op = self.optimizer.apply_gradients(grads)
        self.total_loss = loss
        self.losses = losses
        self.grads = grads
        self.update_op = update_op
        self.args_in = args
        self.kwargs_in = kwargs
        self.built = True

    def _get_feed_dict(self, *args, **kwargs) -> Dict[tf.placeholder, Any]:
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
        assert self._built, "Cannot call update without setting up placeholders."

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


class DatasetGraph(AbstractGraph):

    def __init__(self, 
                 optimizer: tf.train.Optimizer, 
                 loss_function: Callable,
                 grads_function: Callable,
                 learning_rate: float = 1e-3) -> None:
        super(DatasetGraph, self).__init__(optimizer, loss_function, grads_function, learning_rate)
        self._built: bool = False

    def build(self, dataset: tf.data.Dataset) -> None:
        """Sets up dataset handles so that you can call update or loss and just pass in the iterator handle.
        
        Args:
            dataset (tf.data.Dataset): A dataset with appropriate output_types shapes that you plan on training with
        """
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
        batch = iterator.get_next()

        if isinstance(batch, dict):
            grads, loss_packed = self.grads_function(**batch)
        elif isinstance(batch, list) or isinstance(batch, tuple):
            grads, loss_packed = self.grads_function(*batch)
        else:
            grads, loss_packed = self.grads_function(batch)

        loss, losses = self._unpack_losses(loss_packed)
        update_op = self.optimizer.apply_gradients(grads)

        self.total_loss = loss
        self.losses = losses
        self.grads = grads
        self.update_op = update_op
        self.handle = handle

        self._built = True

    def _run_tensor(self, ops: Union[tf.Tensor, Sequence[tf.Tensor]], data_handle: bytes) -> Any:
        """Runs ops on the model with a tf.data.Dataset in graph mode.
        
        Args:
            ops (Union[tf.Tensor, Sequence[tf.Tensor]]): Ops to run on the graph
            data_handle (bytes): Handle to a tf.data.Iterator
        
        Returns:
            Result of ops
        
        Raises:
            RuntimeError: If not run inside a tf Session context
            TypeError: If input data handle is not a bytes object
        """
        if not isinstance(data_handle, bytes):
            raise TypeError("Data handle must be a bytes object")

        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")

        result = sess.run(ops, feed_dict={self.handle: data_handle})
        return result

    def run(self, ops: Union[str, Sequence[tf.Tensor]], *args, **kwargs) -> Any:
        if ops == 'update':
            return self.update(*args, **kwargs)
        elif ops == 'loss':
            return self.loss(*args, **kwargs)
        else:
            return self._run_tensor(ops, *args, **kwargs)

    def update(self, data_handle: bytes) -> Union[float, Tuple]:  # type: ignore
        """Updates the model with a tf.data.Dataset in graph mode.
        
        Args:
            data_handle (bytes): Handle to a tf.data.Iterator
        
        Returns:
            loss (Union[float, Tuple]): Model loss on input batch
        
        Raises:
            RuntimeError: If not run inside a tf Session context
            TypeError: If input data handle is not a bytes object
        """
        _, loss = self._run_tensor([self.update_op, self.losses], data_handle)
        return loss

    def loss(self, data_handle: bytes) -> Union[float, Tuple]:  # type: ignore
        """Gets loss of the model with a tf.data.Dataset in graph mode.
        
        Args:
            data_handle (bytes): Handle to a tf.data.Iterator
        
        Returns:
            loss (Union[float, Tuple]): Model loss on input batch
        
        Raises:
            RuntimeError: If not run inside a tf Session context
            TypeError: If input data handle is not a bytes object
        """
        loss = self._run_tensor(self.losses, data_handle)
        return loss
