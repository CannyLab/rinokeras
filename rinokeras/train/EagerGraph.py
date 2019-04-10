from typing import Union, Callable, Tuple, Sequence, Optional

import tensorflow as tf
import tensorflow.keras.backend as K

from .RinokerasGraph import RinokerasGraph


class EagerGraph(RinokerasGraph):

    def __init__(self,
                 optimizer: tf.train.Optimizer,
                 loss_function: Callable,
                 grads_function: Callable,
                 *args,
                 **kwargs) -> None:
        assert tf.executing_eagerly(), "Cannot use EagerGraph when not in tf eager mode."
        super(EagerGraph, self).__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.grads_function = grads_function
        self._default_operation: Optional[str] = None

    def run(self, ops: Union[str, tf.Tensor, Sequence[tf.Tensor]], *args, **kwargs) -> Union[tf.Tensor, Tuple]:
        if ops == 'default':
            if self._default_operation is None:
                raise ValueError("No default operation in set.")
            ops = self._default_operation

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

    @property
    def default_operation(self) -> Optional[str]:
        return self._default_operation

    @default_operation.setter
    def default_operation(self, value: Optional[str]) -> None:
        if value not in ['update', 'loss', None]:
            raise ValueError(f"Must be one of <update, loss, None>. Received {value}")
        self._default_operation = value
