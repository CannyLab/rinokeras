"""
Dropout-style layers.
"""

import tensorflow as tf
from typing import Dict
from tensorflow.keras import Model # pylint: disable=F0401
import tensorflow.keras.backend as K  # pylint: disable=F0401


class LayerDropout(Model):
    """
    Optionally drops a full layer. Output is x with probability rate and f(x) with probability (1 - rate).

    Args:
        layer_call (Callable[[], Any]): Function that returns output of layer on inputs
        inputs (Any): What to return if the layer is dropped
        rate (float): Rate at which to drop layers

    Returns:
        Any: Either inputs or output of layer_call function.
    """

    def __init__(self, rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rate = rate

    def call(self, layer_outputs, layer_inputs, training=None, **kwargs):
        if training is None:
            training = K.learning_phase()

        output = K.in_train_phase(
            K.switch(K.random_uniform([]) > self.rate, layer_outputs, layer_inputs),
            layer_outputs,
            training=training)
        return output

    def get_config(self) -> Dict:
        config = {
            'rate': self.rate
        }

        return config
