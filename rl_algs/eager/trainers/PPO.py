from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from .PolicyGradient import PGTrainer

class PPOTrainer(PGTrainer):
    
    def __init__(self, obs_shape, ac_shape, policy, discrete, valuecoeff=0.5, entcoeff=0.1, max_grad_norm=0.5, epsilon=0.2, scope='trainer'):
        self._epsilon = epsilon
        self._old_policy = policy.make_copy()
        super().__init__(obs_shape, ac_shape, policy, discrete, valuecoeff, entcoeff, max_grad_norm, scope)

    def _loss_function(self, obs, act, val):
        logits, vpred = self._policy(obs, is_training=True)
        old_logits, old_vpred = self._old_policy(obs, is_training=True)

        neg_logp_actions = self._policy.get_neg_logp_actions(logits, act)
        old_neg_logp_actions = self._old_policy.get_neg_logp_actions(old_logits, act)
        values, advantages = self._compute_values_and_advantages(val, vpred)

        # PPO Surrogate (https://github.com/openai/baselines/blob/master/baselines/ppo2/)
        # Note the order of subtraction. If PPO seems unstable it's probably a function of this being bad
        old_vpred = tf.stop_gradient(old_vpred)
        vpredclipped = old_vpred + tf.clip_by_value(vpred - old_vpred, -self._epsilon, self._epsilon)
        vf_losses1 = tf.square(vpred - values)
        vf_losses2 = tf.square(vpredclipped - values)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(tf.stop_gradient(old_neg_logp_actions) - neg_logp_actions)
        surr1 = -advantages * ratio
        surr2 = -advantages * tf.clip_by_value(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon)
        surr_loss = tf.reduce_mean(tf.maximum(surr1, surr2))

        # Value Loss
        entropy = tf.reduce_mean(self._policy.entropy(logits))

        self._surr_loss = surr_loss
        self._value_loss = value_loss
        self._entropy_loss = entropy

        return surr_loss - self._entcoeff * entropy + self._valuecoeff * value_loss

    def train(self, batch, learning_rate, n_iters=10):
        self._old_policy.set_weights(self._policy.get_weights())
        for key, val in batch.items():
            if val.dtype == np.float64:
                batch[key] = np.asarray(val, dtype=np.float32)
        for _ in range(n_iters):
            grads = self._grads_function(batch['obs'], batch['act'], batch['val'])
            self._optimizer.apply_gradients(zip(grads, self._policy.variables))
        self._num_param_updates += 1
        return self._surr_loss.numpy(), self._value_loss.numpy()
