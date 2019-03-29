from typing import Union, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.contrib.distribute import DistributionStrategy, OneDeviceStrategy

from .PolicyGradient import PolicyGradient

import rinokeras.core.v1x as rk


class PPO(PolicyGradient):

    def __init__(self,
                 model: Model,
                 valuecoeff: float = 0.5,
                 entcoeff: float = 0.0,
                 epsilon: float = 0.2,
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-3,
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Union[float, Tuple[float, float]] = (-1, 1),
                 return_loss_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 distribution_strategy: DistributionStrategy = OneDeviceStrategy('/gpu:0'),
                 old_model: Optional[Model] = None) -> None:

        super().__init__(
            model, valuecoeff, entcoeff, optimizer, learning_rate, gradient_clipping,
            gradient_clipping_bounds, return_loss_summaries, return_grad_summaries, distribution_strategy)

        self.epsilon = epsilon
        self.old_model = old_model

    def build_model(self, inputs):
        obs, act, val, seqlens = inputs
        logits, value = self.model(obs, training=True)

        if self.old_model is None:
            self.old_model = self.model.clone()
        old_logits, old_value = self.old_model(obs, training=True)
        old_logits = tf.stop_gradient(old_logits)
        old_value = tf.stop_gradient(old_value)

        return (logits, value), (old_logits, old_value)

    def loss_function(self, inputs, outputs):
        obs, act, val, seqlens = inputs
        if obs.shape.as_list()[1] is None:
            sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(obs, seqlens)
            sequence_mask = tf.cast(sequence_mask, tf.float32)
        else:
            sequence_mask = 1.0

        (logits, pred_val), (old_logits, old_pred_val) = outputs

        neg_logp_actions = -self.model.logp_actions(logits, act)
        old_neg_logp_actions = -self.old_model.logp_actions(old_logits, act)
        values, advantages = self._compute_values_and_advantages(val, pred_val, sequence_mask)

        # PPO Surrogate (https://github.com/openai/baselines/blob/master/baselines/ppo2/)
        # Note the order of subtraction. If PPO seems unstable it's probably a function of this being bad
        ratio = tf.exp(old_neg_logp_actions - neg_logp_actions)
        surr1 = -advantages * ratio
        surr2 = -advantages * tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        surr_loss = tf.reduce_mean(tf.maximum(surr1, surr2) * sequence_mask)

        # PPO Surrogate value loss
        vpredclipped = old_pred_val + tf.clip_by_value(pred_val - old_pred_val, - self.epsilon, self.epsilon)
        vf_losses1 = tf.square(pred_val - values)
        vf_losses2 = tf.square(vpredclipped - values)
        value_loss = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2) * sequence_mask)

        # Entropy Penalty
        entropy = tf.reduce_mean(self.model.entropy(logits) * sequence_mask)

        metrics = {'Surrogate Loss': surr_loss, 'Value Loss': value_loss, 'Entropy': entropy}

        return surr_loss - self.entcoeff * entropy + self.valuecoeff * value_loss, metrics

    def update_old_model(self) -> None:
        self.old_model.set_weights(self.model.get_weights())
