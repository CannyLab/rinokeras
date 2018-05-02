import re

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .Policy import Policy

class DQNAgent(Policy):
    def __init__(self, obs_shape, num_actions, scope='agent'):
        super().__init__(obs_shape, num_actions)
        self._train_setup = False

        with tf.variable_scope(scope):
            self._scope = tf.get_variable_scope() # do it like this because you could be inside another scope - this will give you the full scope path
            self.sy_obs = tf.placeholder(tf.uint8, [None] + list(obs_shape), name='obs_placeholder')
            self._action, self._qval, self._activ, self._policy_scope = self._setup_agent(self.sy_obs, 'policy')
            self._model_vars = self._policy_scope.global_variables()

    def _setup_agent(self, img_in, scope):
        with tf.variable_scope(scope):
            embedding = self._embedding_network(img_in)
            qval, activ = self._q_network(embedding)
            action = tf.argmax(qval, 1)
            return action, qval, activ, tf.get_variable_scope()

    def _embedding_network(self, img_in, network_architecture=None, reuse=False):
        img_in = tf.cast(img_in, tf.float32) / 255.0

        if network_architecture is None:
            network_architecture = [
                # (32, (8, 8), 4),
                (32, (3, 3), 2),
                (64, (3, 3), 1)
            ]

        with tf.variable_scope('embedding', reuse=reuse):
            embedding = slim.stack(img_in, slim.conv2d, network_architecture)
            return tf.contrib.layers.flatten(embedding)

    def _q_network(self, embedding, network_architecture=None, reuse=False):
        if network_architecture is None:
            network_architecture = [
                        (256, tf.nn.relu),
                        (256, tf.nn.relu)
                    ]

        with tf.variable_scope('qvals', reuse=reuse):
            hidden = slim.stack(embedding, slim.fully_connected, network_architecture)
            qvals = slim.fully_connected(hidden, self._num_actions, activation_fn=None)
            return qvals, hidden

    def _setup_training_placeholders(self):
        self.sy_act = tf.placeholder(tf.int32, [None], name='act_placeholder')
        self.sy_rew = tf.placeholder(tf.float32, [None], name='rew_placeholder')
        self.sy_obs_tp1 = tf.placeholder(tf.uint8, [None] + list(self._obs_shape), name='obs_tp1_placeholder')
        self.sy_done = tf.placeholder(tf.float32, [None], name='done_mask_placeholder')
        self.learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')

    def _setup_loss(self, gamma, target_qval):
        batch_size = tf.shape(self.sy_act)[0]
        indices = tf.stack((tf.range(batch_size), self.sy_act), axis=1)

        # curr_val = Q(s_t, a_t)
        curr_val = tf.gather_nd(self._qval, indices)

        # Q(s_t, a_t) = r_t + max_a Q(s_{t+1}, a)
        target_val = self.sy_rew + gamma * tf.reduce_max(target_qval, axis=1) * (1 - self.sy_done)
        loss = tf.losses.mean_squared_error(labels=target_val, predictions=curr_val)
        tderr = tf.reduce_mean(tf.abs(curr_val - target_val)) # tderr is l1 loss

        return loss, tderr

    def setup_for_training(self, gamma):
        self._train_setup = True
        with tf.variable_scope(self._scope):
            self._setup_training_placeholders()
            _, target_qval, _, target_scope = self._setup_agent(self.sy_obs_tp1, 'target_policy')

            with tf.variable_scope('training'):
                loss, tderr = self._setup_loss(gamma, target_qval)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, **dict(epsilon=1e-4))
                update_op = optimizer.minimize(loss, var_list=self._model_vars)
                update_target_fn = [vt.assign(v) for v, vt in zip(self._model_vars, target_scope.global_variables())]

                self._loss = loss
                self._update_op = update_op
                self._update_target_fn = tf.group(*update_target_fn)

    def _get_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("No default session found. Run this within a tf.Session context.")
        return sess

    def predict(self, obs, return_activations=False):
        sess = self._get_session()
        to_return = self._action if not return_activations else [self._action, self._qval, self._activ]
        to_return = sess.run(to_return, feed_dict={self.sy_obs : obs})
        return to_return[0] if not return_activations else to_return

    def train(self, batch, learning_rate=1e-4):
        if not self._train_setup:
            self.setup_for_training()
        feed_dict = {
            self.sy_obs : batch['obs'],
            self.sy_act : batch['act'],
            self.sy_rew : batch['rew'],
            self.sy_obs_tp1 : batch['obs_tp1'],
            self.sy_done : batch['done'],
            self.learning_rate : learning_rate
        }

        sess = self._get_session()
        _, loss = sess.run([self._update_op, self._loss], feed_dict=feed_dict)
        return loss

    def update_target_network(self):
        sess = self._get_session()
        sess.run(self._update_target_fn)

    def save_model(self, filename, global_step=None):
        sess = self._get_session()
        saver = tf.train.Saver(self._model_vars, filename=filename)
        saver.save(sess, filename, global_step=global_step)

    def load_model(self, filename):
        sess = self._get_session()
        saver = tf.train.Saver(self._model_vars, filename=filename)
        saver.restore(sess, filename)
        if self._train_setup:
            self.update_target_network()

    @property
    def scope(self):
        return self._scope

    @property
    def vars(self):
        return self._model_vars
