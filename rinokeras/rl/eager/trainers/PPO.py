import tensorflow as tf

from .PolicyGradient import PGTrainer

class PPOTrainer(PGTrainer):
    
    def __init__(self, 
                 model, 
                 valuecoeff: float = 0.5, 
                 entcoeff: float = 0.1, 
                 max_grad_norm: float = 0.5, 
                 epsilon: float = 0.2, 
                 scope: str = 'trainer') -> None:
        self._epsilon = epsilon
        self._old_model = model.make_copy()
        super().__init__(model, valuecoeff, entcoeff, max_grad_norm, scope)

    def loss_function(self, obs, act, val):
        logits, vpred = self._model(obs, is_training=True)
        old_logits, old_vpred = self._old_model(obs, is_training=True)

        neg_logp_actions = self._model.get_neg_logp_actions(logits, act)
        old_neg_logp_actions = self._old_model.get_neg_logp_actions(old_logits, act)
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
        entropy = tf.reduce_mean(self._model.entropy(logits))

        self._surr_loss = surr_loss
        self._value_loss = value_loss
        self._entropy_loss = entropy

        return surr_loss - self._entcoeff * entropy + self._valuecoeff * value_loss

    def train(self, batch, learning_rate, n_iters=10):
        self._old_model.set_weights(self._model.get_weights())
        for _ in range(n_iters):
            loss = self._train_on_batch(batch['obs'], batch['act'], batch['val'], learning_rate=learning_rate)

        self._num_param_updates += 1
        
        return [l.numpy() for l in loss]
