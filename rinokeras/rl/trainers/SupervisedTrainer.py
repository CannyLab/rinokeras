import tensorflow as tf

from .Trainer import Trainer

class SupervisedTrainer(Trainer):

    def __init__(self, model, loss_type: str, optimizer: str = 'adam') -> None:
        super().__init__(model, optimizer)
        self._loss_fn = tf.keras.losses.get(loss_type)

    def loss_function(self, features, labels):
        predictions = self._model(features)
        return tf.reduce_mean(self._loss_fn(y_true=labels, y_pred=predictions))

    def train(self, batch, learning_rate=1e-3):
        loss = self._train_on_batch(batch['features'], batch['labels'], learning_rate=learning_rate)
        self._num_param_updates += 1
        return loss
