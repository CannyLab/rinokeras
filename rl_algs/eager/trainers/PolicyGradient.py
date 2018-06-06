from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from .Trainer import Trainer

class PGTrainer(Trainer):
    def __init__(self, obs_shape, ac_shape, policy, discrete, valuecoeff=0.5, entcoeff=0.1, max_grad_norm=0.5, scope='trainer'):
        super().__init__(obs_shape, ac_shape, policy, discrete)
        self._valuecoeff = valuecoeff
        self._entcoeff = entcoeff

    def _compute_values_and_advantages(self, values, pred_values):
        baseline = tf.stop_gradient(pred_values)
        mean, var = tf.nn.moments(baseline, [0])
        self._batch_norm(baseline, mean, var)

        values = tf.constant(values)
        mean, var = tf.nn.moments(values, [0])
        baseline = baseline * (tf.sqrt(var) + 1e-10)
        baseline = baseline + mean

        normed_values = self._batch_norm(values, mean, var)

        advantages = values - baseline
        mean, var = tf.nn.moments(advantages, [0])
        advantages = self._batch_norm(advantages, mean, var)
        return normed_values, advantages
        
    
    def _loss_function(self, obs, act, val):
        logits, pred_values = self._policy(obs, is_training=True)

        neg_logp_actions = self._policy.get_neg_logp_actions(logits, act)
        values, advantages = self._compute_values_and_advantages(val, pred_values)

        # Regular PG Loss
        loss = tf.reduce_mean(advantages * neg_logp_actions)
        # Value Loss
        value_loss = tf.losses.mean_squared_error(labels=values, predictions=pred_values)
        # Entropy Penalty
        entropy = tf.reduce_mean(self._policy.entropy(logits))

        return loss - self._entcoeff * entropy + self._valuecoeff * value_loss, loss, value_loss, entropy

    def train(self, batch, learning_rate):
        loss = self._train_on_batch(batch['obs'], batch['act'], batch['val'])
        self._num_param_updates += 1

        return [l.numpy() for l in loss]





    
