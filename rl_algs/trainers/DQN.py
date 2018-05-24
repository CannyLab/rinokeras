import re

import numpy as np
import tensorflow as tf

from .Trainer import Trainer

class DQNTrainer(Trainer):

    def __init__(self, obs_shape, ac_shape, policy, discrete, gamma=0.99, target_update_freq=10000, scope='trainer'):
        super().__init__(obs_shape, ac_shape, policy, discrete)
        self._target_update_freq = target_update_freq
        self._gamma = gamma
        self._train_vars = self._policy.vars

        with tf.variable_scope(scope):
            self._old_policy = policy.make_copy('old_policy')
            self._scope = tf.get_variable_scope()
            self._setup_placeholders()
            loss, tderr = self._setup_loss()

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            update_op = optimizer.minimize(loss, var_list=self._train_vars)
            update_target_fn = [vt.assign(v) for v, vt in zip(self._policy.vars, self._old_policy.vars)]
            # self._old_policy.create_copy_op_other_to_self(self._policy)

            self._loss = loss
            self._update_op = update_op
            self._update_target_fn = tf.group(*update_target_fn)

    def _setup_placeholders(self):
        self.sy_act = tf.placeholder(tf.int32, [None], name='act_placeholder')
        self.sy_rew = tf.placeholder(tf.float32, [None], name='rew_placeholder')
        self.sy_done = tf.placeholder(tf.float32, [None], name='done_mask_placeholder')
        self.learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')

    def _setup_loss(self):
        batch_size = tf.shape(self.sy_act)[0]
        indices = tf.stack((tf.range(batch_size), self.sy_act), axis=1)

        # curr_val = Q(s_t, a_t)
        curr_val = tf.gather_nd(self._policy.logits, indices)

        # Q(s_t, a_t) = r_t + max_a Q(s_{t+1}, a)
        target_val = self.sy_rew + self._gamma * tf.reduce_max(self._old_policy.logits, axis=1) * (1 - self.sy_done)
        loss = tf.losses.huber_loss(labels=target_val, predictions=curr_val)
        tderr = tf.reduce_mean(tf.abs(curr_val - target_val)) # tderr is l1 loss

        return loss, tderr

    def train(self, batch, learning_rate=1e-4):
        feed_dict = {
            self._policy.sy_obs : batch['obs'],
            self.sy_act : batch['act'],
            self.sy_rew : batch['rew'],
            self._old_policy.sy_obs : batch['obs_tp1'],
            self.sy_done : batch['done'],
            self.learning_rate : learning_rate
        }

        feed_dict.update(self._policy.feed_dict_extras(batch))
        feed_dict.update(self._old_policy.feed_dict_extras(batch))

        sess = self._get_session()
        _, loss, pl, opl = sess.run([self._update_op, self._loss, self._policy.logits, self._old_policy.logits], feed_dict=feed_dict)
        self._num_param_updates += 1
        if self._num_param_updates % self._target_update_freq == 0:
            self.update_target_network()
        return loss

    def update_target_network(self):
        sess = self._get_session()
        sess.run(self._update_target_fn)
        # self._old_policy.copy_other_to_self()
        # self._policy.update_target_network()
