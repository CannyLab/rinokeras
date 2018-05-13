class Trainer(object):

    def __init__(self, obs_shape, ac_shape, policy, discrete=True):
        self._obs_shape = obs_shape
        self._ac_shape = ac_shape
        self._policy = policy
        self._discrete = discrete
        self._scope = None

    def _setup_placeholders(self):
        return NotImplemented

    def _setup_loss(self):
        return NotImplemented

    def _get_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("No default session found. Run this within a tf.Session context.")
        return sess

    def train(self, batch, learning_rate):
        return NotImplemented

    @property
    def scope(self):
        return self._scope