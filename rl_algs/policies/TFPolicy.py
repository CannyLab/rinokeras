import tensorflow as tf

from .Policy import Policy

class TFPolicy(Policy):

    def __init__(self, obs_shape, ac_shape, discrete, scope):
        super().__init__(obs_shape, ac_shape, discrete)
        self._scope_name = scope
        self._scope = None
        self._model_vars = None
        self._logits = None
        self._action = None
        self._value = None

    def _setup_embedding_network(self):
        return NotImplemented

    def _setup_action_logits(self):
        return NotImplemented

    def _setup_value_function(self):
        return NotImplemented

    def _get_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("No default session found. Run this within a tf.Session context.")
        return sess

    def create_copy_op_other_to_self(self, other):
        update_fn = [vs.assign(vo) for vs, vo in zip(self.vars, other.vars)]
        self._copy_other_to_self = tf.group(*update_fn)

    def create_copy_op_self_to_other(self, other):
        update_fn = [vo.assign(vs) for vs, vo in zip(self.vars, other.vars)]
        self._copy_self_to_other = tf.group(*update_fn)

    def copy_other_to_self(self):
        sess = self._get_session()
        sess.run(self._copy_other_to_self)

    def copy_self_to_other(self):
        sess = self._get_session()
        sess.run(self._copy_self_to_other)

    def save_model(self, filename, global_step=None):
        sess = self._get_session()
        # TODO: generic save (don't include scope name)
        saver = tf.train.Saver(self._model_vars, filename=filename)
        saver.save(sess, filename, global_step=global_step)

    def load_model(self, filename):
        sess = self._get_session()
        saver = tf.train.Saver(self._model_vars, filename=filename)
        saver.restore(sess, filename)

    @property
    def scope(self):
        return self._scope

    @property
    def vars(self):
        return self._model_vars

    @property
    def logits(self):
        return self._logits

    @property
    def value(self):
        return self._value


