class Policy(object):
	def __init__(self, obs_shape, num_actions):
		self._obs_shape = obs_shape
		self._num_actions = num_actions

	def setup_for_training(self):
		return NotImplemented

	# Should return the action to perform when seeing obs
	def predict(self, obs):
		return NotImplemented

	# Function to train the policy on a batch
	def train(self, batch, learning_rate):
		return NotImplemented

	# Save the model
	def save_model(self, filename):
		return NotImplemented

	# Load the model
	def load_model(self, filename):
		return NotImplemented

	@property
	def num_actions(self):
		return self._num_actions

	@property
	def obs_shape(self):
		return self._obs_shape