import tensorflow as tf

from .Trainer import Trainer

class SupervisedTrainer(Trainer):

	def __init__(self, model, discrete, optimizer='adam', loss_type=None):
		super().__init__(model, discrete, optimizer)
		if loss_type is None:
			loss_type = 'mse' if not discrete else 'categorical_crossentropy'
		self._loss_fn = tf.keras.losses.get(loss_type)

	def loss_function(self, features, labels):
		predictions = self._model(features)
		return self._loss_fn(y_true=labels, y_pred=predictions)

	def train(self, batch, learning_rate):
		loss = self._train_on_batch(batch['features'], batch['labels'], learning_rate)
		self._num_param_updates += 1
		return loss