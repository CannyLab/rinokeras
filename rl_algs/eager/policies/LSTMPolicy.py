import tensorflow as tf

from rl_algs.eager.common.rnn import EagerLSTM
from .StandardPolicy import StandardPolicy

class LSTMPolicy(StandardPolicy):

    def __init__(self,
                 obs_shape,
                 ac_shape,
                 discrete,
                 action_method='greedy',
                 use_conv=False,
                 embedding_architecture=[64, 64],
                 logit_architecture=[64, 64],
                 value_architecture=[64],
                 lstm_cell_size=256,
                 use_reward=False,
                 initial_logstd=0):

        self._lstm_cell_size = lstm_cell_size
        self._use_reward = use_reward
        super().__init__(obs_shape, ac_shape, discrete, action_method, use_conv,
                         embedding_architecture, logit_architecture,
                         value_architecture, initial_logstd)

    def _setup_agent(self):
        super()._setup_agent()
        self._memory_function = self._setup_memory_function()

    def _setup_memory_function(self):
        if tf.executing_eagerly():
            memory_func = EagerLSTM(self._lstm_cell_size,
                                    return_sequences=True,
                                    return_state=True)
        else:
            memory_func = tf.keras.layers.LSTM(self._lstm_cell_size,
                                               return_sequences=True,
                                               return_state=True)
        return memory_func

    def build(self):
        if not self.built:
            dummy_obs = tf.zeros((1, 1) + self._obs_shape, dtype=tf.float32)
            dummy_rew = tf.zeros((1, 1, 1))

            embedding = self._embedding_function(dummy_obs)
            if self._use_reward:
                embedding = tf.concat((embedding, dummy_rew), -1)
            memory_embed, memory_h, memory_c = self._memory_function(embedding)
            logits = self._logits_function(memory_embed)
            self._value_function(memory_embed)
            self._action_function(logits)
            self.built = True

    def call(self, obs, is_training=False):
        if self._use_reward:
            obs, rew = obs
            # rew = np.asarray(rew, dtype=np.float32)
        # if obs.dtype == np.uint8:
            # obs = np.asarray(obs, np.float32) / 255
        # else:
            # obs = np.asarray(obs, np.float32)
        
        ob_dim = len(self._obs_shape)
        if obs.ndim == ob_dim:
            obs = obs[None, None]
        elif obs.ndim == ob_dim + 1:
            obs = obs[None]
        elif obs.ndim != ob_dim + 2:
            raise ValueError("Received observation of incorrect size. Expected {}. Received {}.".format(
                (None, None) + self._obs_shape, obs.shape))

        batch_size, timesteps = obs.shape[:2]
        rew = tf.reshape(rew, (batch_size, timesteps, 1))

        embedding = self._embedding_function(obs)
        if self._use_reward:
            if (rew.ndim == 2):
                rew = tf.expand_dims(rew, 2)
            embedding = tf.concat((embedding, rew), 2)

        # embedding = tf.reshape(embedding, (batch_size, timesteps, embedding.shape[-1]))
        memory_embed, memory_h, memory_c = self._memory_function(embedding, initial_state=self._memory)
        self._memory = (memory_h, memory_c)
        memory_embed = tf.reshape(memory_embed, (batch_size * timesteps, memory_embed.shape[-1]))
            
        logits = self._logits_function(memory_embed)
        logits = tf.reshape(logits, (batch_size, timesteps) + self._ac_shape)
        if is_training:
            value = self._value_function(memory_embed)
            return logits, tf.squeeze(value)
        else:
            action = self._action_function(logits)
            return action.numpy() if tf.executing_eagerly() else action

    def clear_memory(self):
        self._memory = None

    def make_copy(self):
        return self.__class__(self._obs_shape,
                              self._ac_shape,
                              self._discrete,
                              self._action_method,
                              self._use_conv,
                              self._embedding_architecture,
                              self._logit_architecture,
                              self._value_architecture,
                              self._lstm_cell_size,
                              self._use_reward,
                              self._initial_logstd)
