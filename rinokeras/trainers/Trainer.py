"""Summary
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import tensorflow as tf


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

    def __init__(self,
                 model: tf.keras.Model,
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-3,
                 add_model_losses: bool = True,
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Tuple[float, ...] = (-1, 1)) -> None:
        super().__init__()
        self._model = model

        self._num_param_updates: int = 0
        self._num_param_updates_gpu = tf.get_variable('num_param_updates', shape=(), dtype=tf.int32, trainable=False, 
                                                      initializer=tf.zeros_initializer())
        self._add_model_losses = add_model_losses

        if not tf.executing_eagerly():
            self._increment_step = tf.assign(self._num_param_updates_gpu, self._num_param_updates_gpu + 1)
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
        if gradient_clipping == 'none':
            self._use_gradient_clipping = False
        else:
            self._use_gradient_clipping = True
            self._clipping_method = gradient_clipping
            self._clipping_bounds = gradient_clipping_bounds
            legal_clipping_values = ['value', 'norm', 'global_norm', 'average_norm']
            if gradient_clipping not in legal_clipping_values:
                raise ValueError("Unrecognized gradient clipping method: {}. Must be in {}".format(
                    gradient_clipping, str(legal_clipping_values)))

        self._has_placeholders: bool = False
        self._has_dataset_handle: bool = False

    def _batch_norm(self, array, mean, var):
        """Summary
        
        Args:
            array (TYPE): Description
            mean (TYPE): Description
            var (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        array = array - mean
        array = array / (tf.sqrt(var) + 1e-10)
        return array

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
        if self._use_gradient_clipping:
            if self._clipping_method == 'value':
                grads = [(tf.clip_by_value(g[0], self._clipping_bounds[0], self._clipping_bounds[1]), g[1]) 
                         for g in grads]
            elif self._clipping_method == 'norm':
                grads = [(tf.clip_by_norm(g[0], self._clipping_bounds[0]), g[1]) for g in grads]
            elif self._clipping_method == 'global_norm':
                grads = [(tf.clip_by_global_norm(g[0], self._clipping_bounds[0]), g[1]) for g in grads]
            elif self._clipping_method == 'average_norm':
                grads = [(tf.clip_by_average_norm(g[0], self._clipping_bounds[0]), g[1]) for g in grads]

        return grads, loss_packed

    def _unpack_losses(self, losses):
        """Summary
        
        Args:
            losses (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if isinstance(losses, tuple) or isinstance(losses, list):
            total_loss = losses[0]
        else:
            total_loss = losses

        return total_loss, losses

    def _run_eager(self, do_training: bool, *args, **kwargs):
        """Runs the model in eager mode.
        
        Args:
            do_training (bool): Whether to do a backwards pass or not.
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function
        
        Returns:
            loss (Union[float, tf.EagerTensor]): Model loss on input batch
        """
        assert tf.executing_eagerly(), "Cannot call Trainer._train_eager() when not in tf eager mode."
        if do_training:
            grads, loss = self.grads_function(*args, **kwargs)
            self._optimizer.apply_gradients(zip(grads, self._model.variables))
        else:
            loss = self.loss_function(*args, **kwargs)
        return loss

    def _run_graph_placeholders(self, do_training: bool, *args, **kwargs):
        """Runs the model with placeholders in graph mode.
        
        Args:
            do_training (bool): Whether to do a backwards pass or not.
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function
        
        Returns:
            loss (Union[float, tf.EagerTensor]): Model loss on input batch
        
        Raises:
            KeyError: Description
            RuntimeError: Description
            ValueError: Description
        """
        assert self._has_placeholders, "Cannot call Trainer._train_graph_placeholders without setting up placeholders."

        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")

        if len(args) != len(self._args_in):
            raise ValueError("Expected {} positional arguments, but received {}.".format(
                len(self._args_in), len(args))
            )

        if len(kwargs) != len(self._kwargs_in):
            raise ValueError("Expected {} keyword arguments, but received {}.".format(
                len(self._kwargs_in), len(kwargs))
            )

        feed_dict = {}
        for arg_in, arg in zip(self._args_in, args):
            feed_dict[arg_in] = arg
        for kw in self._kwargs_in:
            try:
                feed_dict[self._kwargs_in[kw]] = kwargs[kw]
            except KeyError:
                raise KeyError("Expected keyword argument '{}'".format(kw))

        update_op = self._placeholder_ops['update_op']
        loss = self._placeholder_ops['loss']

        if do_training:
            _, loss = sess.run([update_op, loss], feed_dict=feed_dict)
        else:
            loss = sess.run(loss, feed_dict=feed_dict)
        return loss

    def _run_graph_handle(self, do_training: bool, data_handle: bytes):
        """Runs the model with a tf.data.Dataset in graph mode.
        
        Args:
            do_training (bool): Whether to do a backwards pass or not.
            data_handle (bytes): Handle to a tf.data.Iterator
        
        Returns:
            loss (Union[float, tf.EagerTensor]): Model loss on input batch
        
        Raises:
            RuntimeError: Description
            TypeError: Description
        """
        assert self._has_dataset_handle, "Cannot call Trainer._train_graph_handle without setting up dataset handles."
        if not isinstance(data_handle, bytes):
            raise TypeError("Data handle must be a bytes object")

        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")

        update_op = self._handle_ops['update_op']
        loss = self._handle_ops['loss']

        if do_training:
            _, loss = sess.run([update_op, loss], feed_dict={self._handle: data_handle})
        else:
            loss = sess.run(loss, feed_dict={self._handle: data_handle})
        return loss

    def _run_on_batch(self,
                      do_training: bool,
                      *args,
                      learning_rate: float = 1e-3,
                      input_data_format: Optional[str] = None,
                      **kwargs):
        """Runs the model on an input batch, automatically choosing which method to run with.
        
        Args:
            do_training (bool): Whether to do a backwards pass or just evaluate the loss.
            *args: Positional arguments to loss function or a handle to a tf.data.Iterator
            learning_rate (float, optional): Learning rate for optimizer
            input_data_format (Optional[str], optional): If model set up for placeholder and dataset training,
                                                         allows you to choose the input data format.
            **kwargs: Keyword arguments to loss function
        
        Returns:
            loss (Union[float, tf.EagerTensor]): Model loss on input batch
        
        Raises:
            ValueError: If run in graph mode and no placeholders/handles are set up, or if you specify a type
                        with input_data_format that the trainer is not set up for.
        """
        self._optimizer._lr = learning_rate
        assert input_data_format in [None, 'placeholders', 'handle'], \
            "<input_data_format> must be one of [None, 'placeholders', 'handle']"
        if tf.executing_eagerly():
            loss = self._run_eager(do_training, *args, **kwargs)
        elif self._has_placeholders and input_data_format != 'handle':
            loss = self._run_graph_placeholders(do_training, *args, **kwargs)
        elif self._has_dataset_handle and input_data_format != 'placeholders':
            # This will fail if you pass in the wrong arguments.
            # Not sure if we should catch this error specifically or not.
            loss = self._run_graph_handle(do_training, *args, **kwargs)
        else:
            raise ValueError("Either placeholders/handle not set up, or input_data_format incorrectly specified.")
        return loss

    def train(self, *args, learning_rate: float = 1e-3, **kwargs):
        """Trains model on an input batch. Can specify the learning rate.
        See the class docstring for full usage instructions.
        
        Args:
            *args: Positional arguments to loss function
            learning_rate (float, optional): Learning rate for optimizer
            **kwargs: Keyword arguments to loss function
        
        Returns:
            loss (Union[float, tf.EagerTensor]): Model loss on input batch
        
        Raises:
            RuntimeError: Description
        """
        loss = self._run_on_batch(True, *args, learning_rate=learning_rate, **kwargs)
        self._num_param_updates += 1
        if tf.executing_eagerly():
            self._num_param_updates_gpu = self._num_param_updates_gpu + 1
        else:
            sess = tf.get_default_session()
            if sess is None:
                raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")
            sess.run(self._increment_step)
        return loss

    def loss(self, *args, **kwargs):
        """Evaluates loss function on an input batch. See the class docstring for full usage instructions.
        
        Args:
            *args: Positional arguments to loss function
            **kwargs: Keyword arguments to loss function
        
        Returns:
            loss (Union[float, tf.EagerTensor]): Model loss on input batch
        """
        loss = self._run_on_batch(False, *args, **kwargs)
        return loss

    def setup_from_placeholders(self, *args, **kwargs) -> None:
        """Sets up placeholders so that you can call trainer.train or trainer.loss as if you're in eager mode.
        
        Args:
            *args: Placeholders for positional arguments to loss function
            **kwargs: Placeholders for keyword arguments to loss function
        """

        grads, loss_packed = self.grads_function(*args, **kwargs)
        loss, losses = self._unpack_losses(loss_packed)

        update_op = self._optimizer.apply_gradients(grads)
        self._placeholder_ops = {'loss': loss, 'losses': losses, 'grads': grads, 'update_op': update_op}
        self._args_in = args
        self._kwargs_in = kwargs
        self._has_placeholders = True

    def setup_from_dataset(self, dataset) -> None:
        """Sets up dataset handles so that you can call trainer.train or trainer.loss and just pass in the
        iterator handle.
        
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
        update_op = self._optimizer.apply_gradients(grads)

        self._handle_ops = {'loss': loss, 'losses': losses, 'grads': grads, 'update_op': update_op}
        self._handle = handle
        self._has_dataset_handle = True

    @property
    def num_param_updates(self) -> int:
        """
        Returns:
            int: Number of times the train(...) function has been called.
        """
        return self._num_param_updates
