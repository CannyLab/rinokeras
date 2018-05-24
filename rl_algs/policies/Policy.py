import numpy as np

class Policy(object):
	def __init__(self, obs_shape, ac_shape, discrete):
		self._obs_shape = obs_shape if not np.isscalar(obs_shape) else (obs_shape,)
		self._ac_shape = ac_shape
		self._discrete = discrete
		if discrete and not np.isscalar(ac_shape):
			raise ValueError("ac_shape must be scalar if mode is discrete")
		elif not discrete and np.isscalar(ac_shape):
			self._ac_shape = (ac_shape,)

	def setup_for_training(self):
		return NotImplemented

	# Should return the action to perform when seeing obs
	def predict(self, obs):
		return NotImplemented

	# Should return the value of an observation
	def predict_value(self, obs):
		return NotImplemented

	# Save the model
	def save_model(self, filename):
		return NotImplemented

	# Load the model
	def load_model(self, filename):
		return NotImplemented

	@property
	def ac_shape(self):
		return self._ac_shape

	@property
	def obs_shape(self):
		return self._obs_shape
