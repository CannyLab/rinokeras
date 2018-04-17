import re

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class DQN(object):

    def __init__(self, sess, input_shape, num_actions):
        self.train_setup = False
        self.sess = sess
        self.num_actions = num_actions
        self._setup_placeholders(input_shape)

        self.q_val, self.q_val_scope = self._setup_q_network(self.sy_obs, scope='q_func', reuse=False)
        self.model_vars = self.q_val_scope.global_variables()

     # Placholders for playing pong (not for training)
    def _setup_placeholders(self, input_shape):
        self.sy_obs = tf.placeholder(tf.uint8, [None] + list(input_shape), name='obs_placeholder')
        self.sy_act = tf.placeholder(tf.int32, [None], name='act_placeholder')
        self.sy_exp = tf.placeholder(tf.int32, [None], name='exp_placeholder')
        self.sy_rew = tf.placeholder(tf.float32, [None], name='rew_placeholder')
        self.sy_obs_tp1 = tf.placeholder(tf.uint8, [None] + list(input_shape), name='obs_tp1_placeholder')
        self.sy_done = tf.placeholder(tf.float32, [None], name='done_mask_placeholder')

    def _setup_q_network(self, img_in, scope, reuse=False):
        flatten = tf.contrib.layers.flatten # just for shorthand
        img_in = tf.cast(img_in, tf.float32) / 255.0
        if scope == 'q_func':
            self.activations = []
        with tf.variable_scope(scope, reuse=reuse):
            out = img_in
            with tf.variable_scope('convnet'):
                # out = slim.conv2d(out, 32, (8, 8), 4)
                # self.activations.append(flatten(out))
                out = slim.conv2d(out, 32, (3, 3), 2)
                if scope == 'q_func':
                    self.activations.append(flatten(out))
                out = slim.conv2d(out, 64, (3, 3), 1)
                if scope == 'q_func':
                    self.activations.append(flatten(out))
            out = flatten(out)
            with tf.variable_scope('action_value'):
                out = slim.fully_connected(out, 512)
                if scope == 'q_func':
                    self.activations.append(out)
                out = slim.fully_connected(out, self.num_actions, activation_fn=None)
                if scope == 'q_func':
                    self.activations.append(out)

            return out, tf.get_variable_scope()

    def setup_for_training(self, gamma):
        self.train_setup = True
        self.learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')
        # self.importance_weights = tf.placeholder(tf.float32, [None], name='importance_weights') # for prioritized DQN
        batch_size = tf.shape(self.sy_act)[0]
        indices = tf.stack((tf.range(batch_size), self.sy_act), axis=1)
        curr_val = tf.gather_nd(self.q_val, indices)

        self.target_q_val, target_scope = self._setup_q_network(self.sy_obs_tp1, scope='target_q_func', reuse=False)
        target_val = self.sy_rew + gamma * tf.reduce_max(self.target_q_val, axis=1) * (1 - self.sy_done)
        # error = self.importance_weights * tf.squared_difference(target_val, curr_val)
        # self.error = tf.reduce_mean(error)
        self.error = tf.losses.mean_squared_error(labels=target_val, predictions=curr_val)
        target_q_func_vars = target_scope.global_variables()

        self.curr_val = curr_val
        self.target_val = target_val
        self.tderr = target_val - curr_val

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, **dict(epsilon=1e-4))
        self.update_op = minimize_and_clip(optimizer, self.error, var_list=self.model_vars, clip_val=10)

        update_target_fn = []
        for v, vt in zip(sorted(self.model_vars, key=lambda v: v.name), sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(vt.assign(v))

        self.update_target_fn = tf.group(*update_target_fn)

    def update_target_network(self):
        if not self.train_setup:
            raise RuntimeError('Called update target network but no target network is set up.')

        self.sess.run(self.update_target_fn)

    def predict(self, obs, return_activations=False):
        if not return_activations:
            return self.sess.run(self.q_val, feed_dict={self.sy_obs : obs})
        else:
            return self.sess.run([self.q_val, self.activations], feed_dict={self.sy_obs : obs})

    def train(self, batch, learning_rate):
        feed_dict = {self.sy_obs : batch['obs'],
                        self.sy_act : batch['act'],
                        self.sy_rew : batch['rew'],
                        self.sy_obs_tp1 : batch['obs_tp1'],
                        self.sy_done : batch['done'],
                        self.learning_rate : learning_rate}
        if 'weights' in batch:
            feed_dict[self.importance_weights] = batch['weights']

        _, tderr = self.sess.run([self.update_op, self.tderr], 
                                        feed_dict=feed_dict)
        return tderr

    def get_error(self, obs, act, rew, done, obs_tp1):
        if np.isscalar(act):
            act = np.array([act])
        if np.isscalar(rew):
            rew = np.array([rew])
        if np.isscalar(done):
            done = np.array([done], dtype=np.bool)
        return self.sess.run(self.tderr, feed_dict={self.sy_obs : obs,
                                                    self.sy_act : act,
                                                    self.sy_rew : rew,
                                                    self.sy_done : done,
                                                    self.sy_obs_tp1 : obs_tp1})[0]

    def save_model(self, filename, global_step=None):
        saver = tf.train.Saver(self.model_vars, filename=filename)
        saver.save(self.sess, filename, global_step=global_step)

    def load_model(self, filename):
        saver = tf.train.Saver(self.model_vars, filename=filename)
        saver.restore(self.sess, filename)
        if self.train_setup:
            self.update_target_network()
