from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .Trainer import Trainer

class PGTrainer(Trainer):
    def __init__(self, obs_shape, ac_shape, policy, discrete, valuecoeff=0.5, entcoeff=0.1, max_grad_norm=0.5, scope='trainer'):
        super().__init__(obs_shape, ac_shape, policy, discrete)
        self._valuecoeff = valuecoeff
        self._entcoeff = entcoeff
        self._train_vars = self._policy.vars

        with tf.variable_scope(scope):
            self._scope = tf.get_variable_scope()
            self._setup_placeholders()
            loss = self._setup_loss()

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
            grads = tf.gradients(loss, self._train_vars)
            if max_grad_norm is not None:
                grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, self._train_vars))
            update_op = optimizer.apply_gradients(grads)
                
            self._loss = loss
            self._update_op = update_op


    def _setup_placeholders(self):
        if self._discrete:
            self.sy_act = tf.placeholder(tf.int32, [None], name='act_placeholder')
        else:
            self.sy_act = tf.placeholder(tf.float32, (None,) + tuple(self._ac_shape), name='act_placeholder')
        self.sy_val = tf.placeholder(tf.float32, [None], name='val_placeholder')
        self.learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')

    def _compute_values_and_advantages(self):
        baseline = tf.stop_gradient(self._policy.value)
        mean, var = tf.nn.moments(baseline, [0])
        baseline = baseline - mean
        baseline = baseline / (tf.sqrt(var) + 1e-10)

        mean, var = tf.nn.moments(self.sy_val, [0])
        baseline = baseline * (tf.sqrt(var) + 1e-10)
        baseline = baseline + mean

        values = self.sy_val
        values = values - mean
        values = values / (tf.sqrt(var) + 1e-10)

        advantages = self.sy_val - baseline
        mean, var = tf.nn.moments(advantages, [0])
        advantages = advantages - mean
        advantages = advantages / (tf.sqrt(var) + 1e-10)
        return values, advantages

    def _setup_loss(self):
        values, advantages = self._compute_values_and_advantages()
        # Regular PG Loss
        loss = tf.reduce_mean(advantages * self._policy.get_neg_logp_actions(self.sy_act))
        # Value Loss
        value_loss = tf.losses.mean_squared_error(labels=values, predictions=self._policy.value)
        # Entropy Penalty
        entropy = tf.reduce_mean(self._policy.entropy())

        self._action_loss = loss
        self._value_loss = value_loss
        self._entropy_loss = entropy

        return loss - self._entcoeff * entropy + self._valuecoeff * value_loss

    def train(self, batch, learning_rate=5e-3):
        feed_dict = {
            self._policy.sy_obs : batch['obs'],
            self.sy_act : batch['act'],
            self.sy_val : batch['val'],
            self.learning_rate : learning_rate
        }
        feed_dict.update(self._policy.feed_dict_extras(batch))

        sess = self._get_session()
        _, loss, value_loss, logstd = sess.run([self._update_op, self._action_loss, self._value_loss, self._policy._log_std], feed_dict=feed_dict)
        print(logstd)
        self._num_param_updates += 1
        return loss, value_loss

class PPOTrainer(PGTrainer):

    def __init__(self, obs_shape, ac_shape, policy, discrete, 
                    valuecoeff=0.8, entcoeff=0.1, max_grad_norm=0.5, epsilon=0.2,
                    scope='trainer'):
        self._epsilon = epsilon
        with tf.variable_scope(scope):
            self._old_policy = policy.make_copy('old_policy')
            self._old_policy.create_copy_op_other_to_self(policy)
        super().__init__(obs_shape, ac_shape, policy, discrete, valuecoeff, entcoeff, max_grad_norm, scope)

    def _setup_placeholders(self):
        super()._setup_placeholders()

    def _setup_loss(self):
        values, advantages = self._compute_values_and_advantages()
        # PPO Surrogate (https://github.com/openai/baselines/blob/master/baselines/ppo2/)
        # Note the order of subtraction. If PPO seems unstable it's probably a function of this being bad
        old_vpred = tf.stop_gradient(self._old_policy.value)
        vpred = self._policy.value
        vpredclipped = old_vpred + tf.clip_by_value(vpred - old_vpred, -self._epsilon, self._epsilon)
        vf_losses1 = tf.square(vpred - values)
        vf_losses2 = tf.square(vpredclipped - values)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(tf.stop_gradient(self._old_policy.get_neg_logp_actions(self.sy_act)) \
                        - self._policy.get_neg_logp_actions(self.sy_act))
        surr1 = -advantages * ratio
        surr2 = -advantages * tf.clip_by_value(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon)
        surr_loss = tf.reduce_mean(tf.maximum(surr1, surr2))

        # Value Loss
        entropy = tf.reduce_mean(self._policy.entropy())

        self._surr_loss = surr_loss
        self._value_loss = value_loss
        self._entropy_loss = entropy

        return surr_loss - self._entcoeff * entropy + self._valuecoeff * value_loss

    def train(self, batch, learning_rate=1e-4, n_iters=10):
        self._old_policy.copy_other_to_self()

        feed_dict = {
            self._policy.sy_obs : batch['obs'],
            self._old_policy.sy_obs : batch['obs'],
            self.sy_act : batch['act'],
            self.sy_val : batch['val'],
            self.learning_rate : learning_rate
        }
        
        feed_dict.update(self._policy.feed_dict_extras(batch))
        feed_dict.update(self._old_policy.feed_dict_extras(batch))
        sess = self._get_session() 

        loss = None
        value_loss = None
        for _ in range(n_iters):
            _, loss, value_loss = sess.run([self._update_op, self._surr_loss, self._value_loss], feed_dict=feed_dict)
        self._num_param_updates += 1
        return loss, value_loss
