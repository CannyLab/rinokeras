from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .Trainer import Trainer

class PGTrainer(Trainer):
    def __init__(self, obs_shape, ac_shape, policy, discrete, alpha=0.8, entcoeff=0.001, scope='traner'):
        super().__init__(obs_shape, ac_shape, policy)
        self._alpha = alpha
        self._entcoeff = entcoeff
        self._discrete = discrete

        with tf.variable_scope(scope):
            self._scope = tf.get_variable_scope()
            self._setup_placeholders()
            loss = self._setup_loss()

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            update_op = optimizer.minimize(loss, var_list=self._policy.vars)
                
            self._loss = loss
            self._update_op = update_op


    def _setup_placeholders(self):
        if self._discrete:
            self.sy_act = tf.placeholder(tf.int32, [None], name='act_placeholder')
        else:
            self.sy_act = tf.placeholder(tf.float32, (None,) + tuple(ac_shape), name='act_placeholder')
        self.sy_val = tf.placeholder(tf.float32, [None], name='val_placeholder')
        self.learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')

    def _compute_values_and_advantages(self):
        baseline = self._policy.value
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

    def _entropy(self):
        if self._discrete:
            probs = tf.nn.softmax(self._policy.logits)
            logprobs = tf.log(probs)
            return -tf.reduce_sum(tf.multiply(probs, logprobs), 1)
        else:
            return -self._policy._log_std

    def _setup_loss(self):
        if self._discrete:
            act_neg_logprobs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_act, logits=self._policy.logits)
        else:
            squared_diff = tf.squared_difference(self.sy_act, self._policy.logits)
            norm_diff = squared_diff / (2 * tf.square(tf.exp(self._policy._log_std)))
            neg_logprobs = norm_diff + (tf.log(2 * np.pi * tf.square(tf.exp(self._policy._log_std))) / 2)
            act_neg_logprobs = tf.reduce_sum(neg_logprobs, 1)
        values, advantages = self._compute_values_and_advantages()
        
        # Regular PG Loss
        loss = tf.reduce_mean(tf.multiply(act_neg_logprobs, advantages))
        # Value Loss
        value_loss = tf.losses.mean_squared_error(labels=values, predictions=self._policy.value)
        # Entropy Penalty
        ent_loss = -self._entcoeff*tf.reduce_mean(self._entropy())

        self._action_loss = loss
        self._value_loss = value_loss
        self._ent_loss = ent_loss
        return self._alpha * loss + (1 - self._alpha) * value_loss + ent_loss

    def train(self, batch, learning_rate=5e-3):
        feed_dict = {
            self._policy.sy_obs : batch['obs'],
            self.sy_act : batch['act'],
            self.sy_val : batch['val'],
            self.learning_rate : learning_rate
        }
        
        sess = self._get_session()
        _, loss = sess.run([self._update_op, self._loss], feed_dict=feed_dict)
        self._num_param_updates += 1
        return loss

# class PPOAgent(PGAgent):

#     def __init__(self, obs_shape, num_actions, scope='agent', use_surrogate=True, epsilon=0.2, dtarg=0.03):
#         self._use_surrogate = use_surrogate
#         self._epsilon = epsilon
#         self._dtarg = dtarg
#         super().__init__(obs_shape, num_actions, scope)

#     def _setup_training_placeholders(self):
#         super()._setup_training_placeholders()
#         self._old_pd_device = tf.placeholder(tf.float32, [None, self._num_actions], name='old_pd_placeholder')
#         self._beta = tf.placeholder(tf.float32, (), name='beta')

#     def _get_old_pd(self, obs, act):
#         sess = self._get_session()
#         self._old_pd_host = sess.run(self._old_logp, feed_dict={self.sy_obs : obs, self.sy_act : act})

#     def _setup_loss(self):
#         batch_size = tf.shape(self.sy_act)[0]
#         indices = tf.stack((tf.range(batch_size), self.sy_act), axis=1)
#         act_logprobs = tf.nn.log_softmax(self._logprobs)
#         self._old_logp = act_logprobs # This is copied onto host then passed back into device to iterate
    
#         logp_act = tf.gather_nd(act_logprobs, indices)
#         old_logp_act = tf.gather_nd(self._old_pd_device, indices)

#         values, advantages = self._compute_values_and_advantages()

#         # PPO Surrogate (https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py#L109)
#         ratio = tf.exp(logp_act - old_logp_act)
#         surr1 = ratio * advantages
#         surr2 = tf.clip_by_value(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon) * advantages
#         surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

#         # Adaptive KL Penalty
#         kl = tf.reduce_sum(tf.multiply(tf.exp(self._old_pd_device), act_logprobs - self._old_pd_device), 1)
#         self._expected_kl = tf.reduce_mean(kl)
#         adaptive_loss = tf.reduce_mean(surr1 - self._beta * kl)

#         # Value Loss
#         value_loss = tf.losses.mean_squared_error(labels=values, predictions=self._value)

#         # Entropy Penalty
#         ent_loss = -self._entcoeff * tf.reduce_mean(self._entropy())
        
#         loss = surr_loss if self._use_surrogate else adaptive_loss

#         return self._alpha * loss + (1 - self._alpha) * value_loss + ent_loss

#     def train(self, batch, learning_rate=1e-4, n_iters=10):
#         if not self._train_setup:
#             self.setup_for_training()

#         self._get_old_pd(batch['obs'], batch['act'])
#         beta = 1.0

#         feed_dict = {
#             self.sy_obs : batch['obs'],
#             self.sy_act : batch['act'],
#             self.sy_val : batch['val'],
#             self._old_pd_device : self._old_pd_host,
#             self.learning_rate : learning_rate
#         }

#         sess = self._get_session() 
#         loss = None
#         for _ in range(n_iters):
#             if self._use_surrogate:
#                 _, loss = sess.run([self._update_op, self._loss], feed_dict=feed_dict)
#             else:
#                 feed_dict[self._beta] = beta
#                 _, loss, d = sess.run([self._update_op, self._loss, self._expected_kl], feed_dict=feed_dict)
#                 if (d < self._dtarg / 1.5):
#                     beta /= 2.0
#                 elif (d > self._dtarg * 1.5):
#                     beta *= 2.0
#         return loss





        




        



