from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf

class Trainer(ABC):

    """
    A trainer that automatically applies a loss function to a model. Designed to work in both eager and non-eager mode.

    Has support for using TFDatasets.
    """
    
    def __init__(self, 
                 model: tf.keras.Model, 
                 optimizer: str = 'adam', 
                 learning_rate: float = 1e-3) -> None:
        super().__init__()
        self._model = model

        self._num_param_updates = 0
        if optimizer == 'adam':
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            self._optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError("Unrecognized optimizer. Received {}.".format(optimizer))

        self._has_placeholders = False
        self._has_dataset_handle = False

    def _batch_norm(self, array, mean, var):
        array = array - mean
        array = array / (tf.sqrt(var) + 1e-10)
        return array

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        raise NotImplementedError("Must implement a loss function.")

    def grads_function(self, *args, **kwargs):
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                loss = self.loss_function(*args, **kwargs)
        
            total_loss, losses = self._unpack_losses(loss)
            grads = tape.gradient(total_loss, self._model.variables)

        else:
            loss = self.loss_function(*args, **kwargs)
            total_loss, losses = self._unpack_losses(loss)
            grads = self._optimizer.compute_gradients(total_loss, self._model.variables)

        return grads, losses

    def _unpack_losses(self, losses):
        if isinstance(losses, tuple) or isinstance(losses, list):
            total_loss = losses[0]
        else:
            total_loss = losses

        return total_loss, losses

    def _train_eager(self, *args, **kwargs):
        assert tf.executing_eagerly(), "Cannot call Trainer._train_eager() when not in tf eager mode."
        grads, loss = self.grads_function(*args, **kwargs)
        self._optimizer.apply_gradients(zip(grads, self._model.variables))
        return loss

    def _train_graph_placeholders(self, *args, **kwargs):
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

        _, loss = sess.run([self._update_op, self._loss], feed_dict=feed_dict)
        return loss

    def _train_graph_handle(self, data_handle: bytes):
        assert self._has_dataset_handle, "Cannot call Trainer._train_graph_handle without setting up dataset handles."
        if not isinstance(data_handle, bytes):
            raise TypeError("Data handle must be a bytes object")

        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")

        _, loss = sess.run([self._update_op, self._loss], feed_dict={self._handle: data_handle})
        return loss

    def _train_on_batch(self, 
                        *args, 
                        learning_rate: float = 1e-3, 
                        input_data_format: Optional[str] = None, 
                        **kwargs):
        self._optimizer._lr = learning_rate
        assert input_data_format in [None, 'placeholders', 'handle'], \
            "<input_data_format> must be one of [None, 'placeholders', 'handle']"
        if tf.executing_eagerly():
            loss = self._train_eager(*args, **kwargs)
        elif self._has_placeholders and input_data_format != 'handle':
            loss = self._train_graph_placeholders(*args, **kwargs)
        elif self._has_dataset_handle and input_data_format != 'placeholders':
            # This will fail if you pass in the wrong arguments.
            # Not sure if we should catch this error specifically or not.
            loss = self._train_graph_handle(*args, **kwargs)
        else:
            raise ValueError("Either placeholders/handle not set up, or input_data_format incorrectly specified.")
        return loss

    def train(self, *args, learning_rate: float = 1e-3, **kwargs):
        loss = self._train_on_batch(*args, learning_rate=learning_rate, **kwargs)
        self._num_param_updates += 1
        return loss

    def setup_from_placeholders(self, *args, **kwargs) -> None:
        loss = self.loss_function(*args, **kwargs)
        total_loss, losses = self._unpack_losses(loss)
        update_op = self._optimizer.minimize(total_loss, var_list=self._model.variables)

        self._args_in = args
        self._kwargs_in = kwargs

        self._loss = total_loss
        self._update_op = update_op
        self._has_placeholders = True

    def setup_from_dataset(self, dataset) -> None:
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
        batch = iterator.get_next()
        if isinstance(batch, dict):
            loss = self.loss_function(**batch)
        elif isinstance(batch, list) or isinstance(batch, tuple):
            loss = self.loss_function(*batch)
        else:
            loss = self.loss_function(batch)

        total_loss, losses = self._unpack_losses(loss)
        update_op = self._optimizer.minimize(total_loss, var_list=self._model.variables)

        self._handle = handle
        self._loss = total_loss
        self._update_op = update_op
        self._has_dataset_handle = True

    @property
    def num_param_updates(self):
        return self._num_param_updates
