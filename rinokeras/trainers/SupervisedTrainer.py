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
        self.predictions = tf.argmax(predictions, -1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(
            predictions, -1, output_type=tf.int32), tf.squeeze(labels)), tf.float32))
        # return tf.reduce_mean(self._loss_fn(y_true=labels, y_pred=predictions))
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=predictions)
