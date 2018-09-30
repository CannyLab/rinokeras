import tensorflow as tf
from typing import Tuple, Union
from .Trainer import Trainer


class SupervisedTrainer(Trainer):

    def __init__(self,
                 model: tf.keras.Model,
                 loss_type: str,
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-3,
                 add_model_losses: bool = True,
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Union[float, Tuple[float, ...]] = (-1, 1),
                 num_gpus: int = 1) -> None:
        super().__init__(model, optimizer, learning_rate, add_model_losses, gradient_clipping,
                         gradient_clipping_bounds, num_gpus)
        self._loss_fn = tf.keras.losses.get(loss_type)

    def loss_function(self, features, labels):
        predictions = self._model(features)
        return tf.reduce_mean(self._loss_fn(y_true=labels, y_pred=predictions))
