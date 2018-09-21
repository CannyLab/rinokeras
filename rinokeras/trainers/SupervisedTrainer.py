import tensorflow as tf
from typing import Tuple
from .Trainer import Trainer


class SupervisedTrainer(Trainer):

    def __init__(self, 
                 model, 
                 loss_type: str, 
                 optimizer: str = 'adam',
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Tuple[float, ...] = (-1, 1),
                 **kwargs) -> None:
        super().__init__(model, optimizer,
                         gradient_clipping=gradient_clipping, 
                         gradient_clipping_bounds=gradient_clipping_bounds,
                         **kwargs)
        self._loss_fn = tf.keras.losses.get(loss_type)

    def loss_function(self, features, labels):
        predictions = self._model(features)
        return tf.reduce_mean(self._loss_fn(y_true=labels, y_pred=predictions))
