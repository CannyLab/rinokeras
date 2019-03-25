from typing import Union, Tuple

import tensorflow as tf
from tensorflow.keras import Model

from tensorflow.contrib.distribute import DistributionStrategy, OneDeviceStrategy

from rinokeras.train import Experiment
import rinokeras.v1x as rk


class PolicyGradient(Experiment):
    def __init__(self,
                 model: Model,
                 valuecoeff: float = 0.5,
                 entcoeff: float = 0.0,
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-3,
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Union[float, Tuple[float, float]] = (-1, 1),
                 return_loss_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 distribution_strategy: DistributionStrategy = OneDeviceStrategy('/gpu:0')) -> None:

        super().__init__(
            model, optimizer, learning_rate, gradient_clipping,
            gradient_clipping_bounds, return_loss_summaries, return_grad_summaries, distribution_strategy)

        self.valuecoeff = valuecoeff
        self.entcoeff = entcoeff

    def build_model(self, inputs):
        obs, act, val, seqlens = inputs
        logits, value = self.model(obs, training=True)
        return logits, value

    def _compute_values_and_advantages(self, values, pred_values, sequence_mask):

        def masked_moments(array):
            if sequence_mask in [None, 1.0]:
                mean, variance = tf.nn.moments(array, [0])
            else:
                with tf.control_dependencies([tf.assert_none_equal(tf.reduce_sum(sequence_mask, 0), 0.0)]):
                    mean = tf.reduce_sum(array * sequence_mask, 0) \
                        / tf.reduce_sum(sequence_mask, 0)
                    variance = tf.reduce_sum(
                        tf.squared_difference(array, tf.stop_gradient(mean)) * sequence_mask,
                        0) / tf.reduce_sum(sequence_mask, 0)

            return mean, variance

        def batch_norm(tensor, mean, var):
            return (tensor - mean[None]) / tf.sqrt(var[None] + 1e-10)

        baseline = tf.stop_gradient(pred_values)
        mean, var = masked_moments(baseline)
        batch_norm(baseline, mean, var)

        mean, var = masked_moments(values)
        baseline = baseline * (tf.sqrt(var) + 1e-10)
        baseline = baseline + mean

        normed_values = batch_norm(values, mean, var)

        advantages = values - baseline
        mean, var = masked_moments(advantages)
        advantages = batch_norm(advantages, mean, var)
        return normed_values, advantages

    def loss_function(self, inputs, outputs):
        obs, act, val, seqlens = inputs
        if obs.shape.as_list()[1] is None:
            sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(obs, seqlens)
            sequence_mask = tf.cast(sequence_mask, tf.float32)
        else:
            sequence_mask = 1.0

        logits, pred_val = outputs

        neg_logp_actions = -self.model.logp_actions(logits, act)
        values, advantages = self._compute_values_and_advantages(val, pred_val, sequence_mask)
        # Regular PG Loss
        loss = tf.reduce_mean(advantages * neg_logp_actions * sequence_mask)
        # Value Loss
        value_loss = tf.losses.mean_squared_error(labels=values, predictions=pred_val, weights=sequence_mask)
        # Entropy Penalty
        entropy = tf.reduce_mean(self.model.entropy(logits) * sequence_mask)

        metrics = {'Advanatage Loss': loss, 'Value Loss': value_loss, 'Entropy': entropy}

        return loss - self.entcoeff * entropy + self.valuecoeff * value_loss, metrics
