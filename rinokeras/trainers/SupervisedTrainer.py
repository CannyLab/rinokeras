import tensorflow as tf

from .Trainer import Trainer

class SupervisedTrainer(Trainer):

    def __init__(self, model, loss_type: str, optimizer: str = 'adam') -> None:
        super().__init__(model, optimizer)
        self._loss_fn = tf.keras.losses.get(loss_type)

    def loss_function(self, features, labels):
        predictions = self._model(features)
        return tf.reduce_mean(self._loss_fn(y_true=labels, y_pred=predictions))
