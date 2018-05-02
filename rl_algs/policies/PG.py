import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .Policy import Policy

class PGAgent(Policy):
    def __init__(self, obs_shape, num_actions, scope='agent'):
        super().__init__(obs_shape, num_actions)
        self._train_setup = False

        with tf.variable_scope(scope):
            self._scope = tf.get_variable_scope() # do it like this because you could be inside another scope - this will give you the full scope path
            self.sy_obs = tf.placeholder(tf.uint8, [None] + list(obs_shape), name='obs_placeholder')
            self._action, self._logprobs, self._activ, self._value, self._policy_scope = self._setup_agent(self.sy_obs, 'policy')
            self._model_vars = self._policy_scope.global_variables()

    def _setup_agent(self, img_in, scope):
        with tf.variable_scope(scope):
            embedding = self._embedding_network(img_in)
            logprobs, activ = self._action_network(embedding)
            action = tf.squeeze(tf.multinomial(logprobs, 1)) # remove extraneous dimension
            val = self._value_network(embedding)
            return action, logprobs, activ, val, tf.get_variable_scope()
        
    def _embedding_network(self, img_in, network_architecture=None, reuse=False):
        img_in = tf.cast(img_in, tf.float32) / 255.0

        if network_architecture is None:
            network_architecture = [
                (32, (3, 3), 2),
                (64, (3, 3), 1)
            ]

        with tf.variable_scope('embedding', reuse=reuse):
            embedding = slim.stack(img_in, slim.conv2d, network_architecture)
            return tf.contrib.layers.flatten(embedding)

    def _action_network(self, embedding, network_architecture=None, reuse=False):
        if network_architecture is None:
            network_architecture = [
                        (256, tf.nn.relu),
                        (256, tf.nn.relu)
                    ]

            with tf.variable_scope('action', reuse=reuse):
                hidden = slim.stack(embedding, slim.fully_connected, network_architecture)
                logprobs = slim.fully_connected(hidden, self._num_actions, activation_fn=None)
                return logprobs, hidden

    def _value_network(self, embedding, network_architecture=None, reuse=False):
        if network_architecture is None:
            network_architecture = [
                        (256, tf.nn.relu),
                        (256, tf.nn.relu)
                    ]

            with tf.variable_scope('value', reuse=reuse):
                hidden = slim.stack(embedding, slim.fully_connected, network_architecture)
                value = slim.fully_connected(hidden, 1, activation_fn=None)
                value = tf.squeeze(value) # remove extraneous dimension
                return value

    def _setup_training_placeholders(self):
        self.sy_act = tf.placeholder(tf.int32, [None], name='act_placeholder')
        # self.sy_adv = tf.placeholder(tf.float32, [None], name='adv_placeholder')
        self.sy_val = tf.placeholder(tf.float32, [None], name='val_placeholder')
        self.learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')

    def _compute_values_and_advantages(self):
        baseline = self._value
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
        probs = tf.nn.softmax(self._logprobs)
        logprobs = tf.log(probs)
        return -tf.reduce_sum(tf.multiply(probs, logprobs), 1)

    def _setup_loss(self):
        batch_size = tf.shape(self.sy_act)[0]
        indices = tf.stack((tf.range(batch_size), self.sy_act), axis=1)
        act_logprobs = tf.nn.log_softmax(self._logprobs)
        logp_act = tf.gather_nd(act_logprobs, indices)

        values, advantages = self._compute_values_and_advantages()
        
        # Regular PG Loss
        loss = tf.reduce_mean(-logp_act * advantages)
        # Value Loss
        value_loss = tf.losses.mean_squared_error(labels=values, predictions=self._value)
        # Entropy Penalty
        ent_loss = -self._entcoeff*tf.reduce_mean(self._entropy())

        self.loss = loss
        self.value_loss = value_loss
        self.ent_loss = ent_loss
        return self._alpha * loss + (1 - self._alpha) * value_loss + ent_loss

    def setup_for_training(self, alpha=0.8, entcoeff=0.01):
        self._train_setup = True
        self._alpha = alpha
        self._entcoeff = entcoeff
        with tf.variable_scope(self._scope):
            self._setup_training_placeholders()
            with tf.variable_scope('training'):
                loss = self._setup_loss()

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                update_op = optimizer.minimize(loss, var_list=self._model_vars)
                
                self._loss = loss
                self._update_op = update_op

    def _get_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("No default session found. Run this within a tf.Session context.")
        return sess

    def predict(self, obs, return_activations=False):
        sess = self._get_session()
        to_return = self._action if not return_activations else [self._action, self._value, selv._probs, self._activ]
        to_return = sess.run(to_return, feed_dict={self.sy_obs : obs})
        return to_return

    def train(self, batch, learning_rate=1e-4):
        if not self._train_setup:
            self.setup_for_training()

        feed_dict = {
            self.sy_obs : batch['obs'],
            self.sy_act : batch['act'],
            self.sy_val : batch['val'],
            # self.sy_adv : batch['adv'],
            self.learning_rate : learning_rate
        }
        
        sess = self._get_session()
        _, loss, l1, vf, el = sess.run([self._update_op, self._loss, self.loss, self.value_loss, self.ent_loss], feed_dict=feed_dict)
        print(self._alpha * l1, (1 - self._alpha) * vf, self._entcoeff * el)
        return loss

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

class PPOAgent(PGAgent):

    def __init__(self, obs_shape, num_actions, scope='agent', use_surrogate=True, epsilon=0.2, dtarg=0.03):
        self._use_surrogate = use_surrogate
        self._epsilon = epsilon
        self._dtarg = dtarg
        super().__init__(obs_shape, num_actions, scope)

    def _setup_training_placeholders(self):
        super()._setup_training_placeholders()
        self._old_pd_device = tf.placeholder(tf.float32, [None, self._num_actions], name='old_pd_placeholder')
        self._beta = tf.placeholder(tf.float32, (), name='beta')

    def _get_old_pd(self, obs, act):
        sess = self._get_session()
        self._old_pd_host = sess.run(self._old_logp, feed_dict={self.sy_obs : obs, self.sy_act : act})

    def _setup_loss(self):
        batch_size = tf.shape(self.sy_act)[0]
        indices = tf.stack((tf.range(batch_size), self.sy_act), axis=1)
        act_logprobs = tf.nn.log_softmax(self._logprobs)
        self._old_logp = act_logprobs # This is copied onto host then passed back into device to iterate
    
        logp_act = tf.gather_nd(act_logprobs, indices)
        old_logp_act = tf.gather_nd(self._old_pd_device, indices)

        values, advantages = self._compute_values_and_advantages()

        # PPO Surrogate (https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py#L109)
        ratio = tf.exp(logp_act - old_logp_act)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon) * advantages
        surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # Adaptive KL Penalty
        kl = tf.reduce_sum(tf.multiply(tf.exp(self._old_pd_device), act_logprobs - self._old_pd_device), 1)
        self._expected_kl = tf.reduce_mean(kl)
        adaptive_loss = tf.reduce_mean(surr1 - self._beta * kl)

        # Value Loss
        value_loss = tf.losses.mean_squared_error(labels=values, predictions=self._value)

        # Entropy Penalty
        ent_loss = -self._entcoeff * tf.reduce_mean(self._entropy())
        
        loss = surr_loss if self._use_surrogate else adaptive_loss

        return self._alpha * loss + (1 - self._alpha) * value_loss + ent_loss

    def train(self, batch, learning_rate=1e-4, n_iters=10):
        if not self._train_setup:
            self.setup_for_training()

        self._get_old_pd(batch['obs'], batch['act'])
        beta = 1.0

        feed_dict = {
            self.sy_obs : batch['obs'],
            self.sy_act : batch['act'],
            self.sy_val : batch['val'],
            self._old_pd_device : self._old_pd_host,
            self.learning_rate : learning_rate
        }

        sess = self._get_session() 
        loss = None
        for _ in range(n_iters):
            if self._use_surrogate:
                _, loss = sess.run([self._update_op, self._loss], feed_dict=feed_dict)
            else:
                feed_dict[self._beta] = beta
                _, loss, d = sess.run([self._update_op, self._loss, self._expected_kl], feed_dict=feed_dict)
                if (d < self._dtarg / 1.5):
                    beta /= 2.0
                elif (d > self._dtarg * 1.5):
                    beta *= 2.0
        return loss





        




        



