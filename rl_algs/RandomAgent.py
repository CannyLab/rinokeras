import numpy as np

from .Policy import Policy

class RandomAgent(Policy):

	def predict(self, obs):
		return np.random.randint(self._num_actions)