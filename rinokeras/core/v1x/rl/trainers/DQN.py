from typing import Union, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import Model

from tensorflow.contrib.distribute import DistributionStrategy, OneDeviceStrategy

from rinokeras.core.v1x.train import Experiment
import rinokeras.core.v1x as rk

class DQN(Experiment):

    def __init__(self,
                 model: Model,
                 target_update_freq: int = 1000,
                 optimizer: str = 'adam',
                 learning_rate: float = 1e-3,
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Union[float, Tuple[float, float]] = (-1, 1),
                 return_loss_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 distribution_strategy: DistributionStrategy = OneDeviceStrategy('/gpu:0'),
                 old_model: Optional[Model] = None) -> None:

        super().__init__(
            model, optimizer, learning_rate, gradient_clipping,
            gradient_clipping_bounds, return_loss_summaries, return_grad_summaries,
            distribution_strategy)

        self.target_update_freq = target_update_freq
        self.old_model = old_model

    def build_model(self, inputs):
        obs, act, rew, nextobs, seqlens = inputs
        logits, value = self.model(obs, training=True)
        if self.old_model is None:
            self.old_model = self.model.clone()
        old_logits, old_value = self.old_model(obs, training=True)
        return (logits, value), (old_logits, old_value)

    def loss_function(self, inputs, outputs):
        obs, act, rew, nextobs, seqlens, done = inputs
        (logits, value), (old_logits, old_value) = outputs

        batch_size = tf.shape(act)[0]
        indices = tf.stack((tf.range(batch_size), act), axis=1)

        # curr_val = Q(s_t, a_t)
        curr_val = tf.gather_nd(logits, indices)

        # Q(s_t, a_t) = r_t + max_a Q(s_{t+1}, a)
        target_val = rew + self.gamma * tf.reduce_max(old_logits, axis=-1) * (1 - done)
        loss = tf.losses.huber_loss(labes=target_val, predictions=curr_val)
        return loss
