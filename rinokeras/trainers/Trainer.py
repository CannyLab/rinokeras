from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Callable, Sequence, List

import tensorflow as tf

from .graphs import EagerGraph, PlaceholderGraph, DatasetGraph


class Trainer(ABC):

    """A trainer that automatically applies a loss function to a model. Designed to work in both eager and non-eager
    mode. This is an abstract class. To use it, subclass it and implement Trainer.loss_function(...). You shouldn't need
    to signficantly alter any other functions (except for certain algorithms like PPO where the loss function gets
    updated as you iterate over a batch).

    Args:
            model (tf.keras.Model): Model to optimize
            optimizer (str, optional): Type of tensorflow optimizer to use
            learning_rate (float, optional): Learning rate for optimizer
            add_model_losses (bool, optional): Whether to add regularization and other losses
            gradient_clipping (str, optional): Type of gradient clipping to use
            gradient_clipping_bounds (Tuple[float, ...], optional): Norm or values to use for gradient clipping
        
    Raises:
        ValueError: If an unrecognized optimizer type is passed in or an unrecognized gradient clip type
    
    Usage:
        Eager Mode:
            The trainer is designed to be extremely easy to use in Eager mode. Once you define the loss
            function, just call Trainer.train(...) or Trainer.loss(...) using the same arguments you defined in
            loss_function(...). As an example, suppose we define the loss function like
    
                def loss_function(features, labels):
                    ...
    
            Then we would call any of the following:
    
                loss = trainer.train(batch_features, batch_labels)
                loss = trainer.train(features=batch_features, labels=batch_labels)
    
                batch = {'features', 'labels'}
                loss = trainer.train(**batch)
    
            In eager mode, Trainer.loss is just an alias to loss_function.
    
        Placeholders:
            In non-eager mode, when training with placeholders, the usage is designed to be extremely similar
            to eager mode. However, instead of calling Trainer.train(...) right away, you have to call
            Trainer.setup_from_placeholders(...). As an example, we would say
    
                feature_ph = tf.placeholder(...)
                label_ph = tf.placeholder(...)
                trainer.setup_from_placeholders(feature_ph, label_ph)
                trainer.train(batch_features, batch_labels)
    
            A few notes. Trainer.setup_from_placeholders(...) only needs to be called once. Also, while you can
            set up placeholders in any way, you must follow the same signature afterwards when you call train.
            In other words, if you pass in placeholders as non-keyword arguments, you must pass in batches
            as non-keyword arguments and vice versa. A TODO is to fix this, possibly using getfullargspec.
    
        Datasets:
            Training with tf datasets is a little different, although arguably easier. To do so, you must
            first call Trainer.setup_from_dataset(<my_dataset>), passing in your tf.data.Dataset. That
            is the only setup required. At train time, get the iterator for your dataset, then get the
            string handle via
    
                data_handle = sess.run(iterator.string_handle())
    
            You then pass this handle in to Trainer.train(...) or Trainer.loss(...), and that's it!
            The only tricky thing about this is that it passes inputs from the dataset to your loss function
            like so:
    
                batch = iterator.get_next()
                loss = trainer.loss_function(*batch) or trainer.loss_function(**batch)
    
            This means that your dataset should parse examples into either a tuple with the same ordering
            as the arguments to loss_function or a dict with the same keys as the loss function has variable names.
    """

    _num_trainers: int = 0

    def __init__(self,
                 model: tf.keras.Model,
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-3,
                 add_model_losses: bool = True,
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Union[float, Tuple[float, ...]] = (-1, 1)) -> None:
        super().__init__()
        self._name = self.__class__.__name__.lower()
        if Trainer._num_trainers > 0:
            self._name += '_{}'.format(Trainer._num_trainers)
        Trainer._num_trainers += 1
        self._model = model

        self._add_model_losses = add_model_losses
        self._num_param_updates: int = 0
        self.learning_rate = learning_rate

        with tf.variable_scope(self._name):
            if optimizer == 'adam':
                self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                self._optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            elif optimizer == 'sgd':
                self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif optimizer == 'momentum':
                self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8)
            else:
                raise ValueError("Unrecognized optimizer. Received {}.".format(optimizer))

        # Setup gradient clipping
        self._clip_gradients = self._get_gradient_clip_function(gradient_clipping, gradient_clipping_bounds)

        if tf.executing_eagerly():
            self._eager_graph = EagerGraph(self._optimizer, self.loss_function, self.grads_function, learning_rate)

        self._has_placeholders: bool = False
        self._has_dataset_handle: bool = False

    def _get_gradient_clip_function(self, clip_type: str, clip_bounds: Union[float, Tuple[float, ...]]) -> \
            Callable[[Sequence], List]:

        def clip_func(grads):
            if clip_type in ['none', 'None']:
                return grads
            grads = []
            for g, v in grads:
                if g is None:
                    grads.append((None, v))
                    continue

                if clip_type == 'value':
                    g = tf.clip_by_value(g, clip_bounds[0], clip_bounds[1])
                elif clip_type == 'norm':
                    g = tf.clip_by_norm(g, clip_bounds)
                elif clip_type == 'global_norm':
                    g = tf.clip_by_global_norm(g, clip_bounds)
                elif clip_type == 'average_norm':
                    g = tf.clip_by_average_norm(g, clip_bounds)
                else:
                    raise ValueError("Unrecognized gradient clipping method: {}.".format(clip_type))
                grads.append((g, v))
            return grads
        return clip_func

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        """Loss function to be implemented by subclasses
        
        Args:
            *args: Positional arguments to loss function
            **kwargs: Keyword Arguments to loss function
        
        Raises:
            NotImplementedError: Trainer is an abstract class, must subclass and implement loss_function
        """
        raise NotImplementedError("Must implement a loss function.")

    def grads_function(self, *args, **kwargs):
        """Computes the gradient of the loss function wrt model parameters.
        
        Args:
            *args: Positional arguments to loss_function
            **kwargs: Key word arguments to loss function
        
        Returns:
            grads (List[tf.Tensor]): Gradients of the model parameters
            losses: Different losses returned by loss_function
        """
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                loss_packed = self.loss_function(*args, **kwargs)
                total_loss, _ = self._unpack_losses(loss_packed)
                loss_to_optimize = total_loss if not self._add_model_losses else total_loss + sum(self._model.losses)

            grads = tape.gradient(loss_to_optimize, self._model.variables)
            grads = zip(grads, self._model.variables)
        else:
            loss_packed = self.loss_function(*args, **kwargs)
            total_loss, _ = self._unpack_losses(loss_packed)
            loss_to_optimize = total_loss if not self._add_model_losses else total_loss + sum(self._model.losses)
            grads = self._optimizer.compute_gradients(loss_to_optimize, self._model.variables)

        # By default all of these norms use L2 TODO: Add additional norm types to the options
        grads = self._clip_gradients(grads)

        return grads, loss_packed

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

    def _run_graph(self,
                   ops: Union[str, Sequence[tf.Tensor]],
                   *args,
                   learning_rate: float = 1e-3,
                   input_data_format: Optional[str] = None,
                   **kwargs):
        """Runs the model on an input batch, automatically choosing which method to run with.
        
        Args:
            ops (Union[str, Sequence[tf.Tensor]]): Ops to run on the graph, or update/loss string
            *args: Positional arguments to loss function or a handle to a tf.data.Iterator
            learning_rate (float, optional): Learning rate for optimizer
            input_data_format (Optional[str], optional): If model set up for placeholder and dataset training,
                                                         allows you to choose the input data format.
            **kwargs: Keyword arguments to loss function
        
        Returns:
            result: Result of running ops on graph
        
        Raises:
            ValueError: If run in graph mode and no placeholders/handles are set up, or if you specify a type
                        with input_data_format that the trainer is not set up for.
        """
        self._optimizer._lr = learning_rate
        assert input_data_format in [None, 'placeholders', 'handle'], \
            "<input_data_format> must be one of [None, 'placeholders', 'handle']"
        if tf.executing_eagerly():
            result = self._eager_graph.run(ops, *args, **kwargs)
        elif self._has_placeholders and input_data_format != 'handle':
            result = self._placeholder_graph.run(ops, *args, **kwargs)
        elif self._has_dataset_handle and input_data_format != 'placeholders':
            # This will fail if you pass in the wrong arguments.
            # Not sure if we should catch this error specifically or not.
            result = self._dataset_graph.run(ops, *args, **kwargs)
        else:
            raise ValueError("Either placeholders/handle not set up, or input_data_format incorrectly specified.")
        return result

    def run(self, ops: Union[tf.Tensor, Sequence[tf.Tensor]], *args, **kwargs):
        """Runs ops on graph. Use this to get the value of any arbitrary tensor, do not use for training.
        
        Args:
            ops (Union[str, Sequence[tf.Tensor]]): Ops to run on the graph
            *args: Positional arguments to loss function
            **kwargs: Keyword arguments to loss function
        
        Returns:
            Result of running ops on graph
        """
        result = self._run_graph(ops, *args, **kwargs)
        return result

    def train(self, *args, learning_rate: float = 1e-3, **kwargs):
        """Trains model on an input batch. Can specify the learning rate.
        See the class docstring for full usage instructions.
        
        Args:
            *args: Positional arguments to loss function
            learning_rate (float, optional): Learning rate for optimizer
            **kwargs: Keyword arguments to loss function
        
        Returns:
            loss (Union[float, tf.Tensor]): Model loss on input batch
        
        Raises:
            RuntimeError: Description
        """
        loss = self._run_graph('update', *args, learning_rate=learning_rate, **kwargs)
        self._num_param_updates += 1
        return loss

    def loss(self, *args, **kwargs):
        """Evaluates loss function on an input batch. See the class docstring for full usage instructions.
        
        Args:
            *args: Positional arguments to loss function
            **kwargs: Keyword arguments to loss function
        
        Returns:
            loss (Union[float, tf.Tensor]): Model loss on input batch
        """
        loss = self._run_graph('loss', *args, **kwargs)
        return loss

    def setup_from_placeholders(self, *args, **kwargs) -> None:
        """Sets up placeholders so that you can call trainer.train or trainer.loss as if you're in eager mode.
        
        Args:
            *args: Placeholders for positional arguments to loss function
            **kwargs: Placeholders for keyword arguments to loss function
        """
        self._placeholder_graph = PlaceholderGraph(
            self._optimizer, self.loss_function, self.grads_function, self.learning_rate)
        self._has_placeholders = True

    def setup_from_dataset(self, dataset) -> None:
        """Sets up dataset handles so that you can call trainer.train or trainer.loss and just pass in the
        iterator handle.
        
        Args:
            dataset (tf.data.Dataset): A dataset with appropriate output_types shapes that you plan on training with
        """
        self._dataset_graph = DatasetGraph(
            self._optimizer, self.loss_function, self.grads_function, self.learning_rate)
        self._has_dataset_handle = True

    @property
    def num_param_updates(self) -> int:
        """
        Returns:
            int: Number of times the train(...) function has been called.
        """
        return self._num_param_updates
