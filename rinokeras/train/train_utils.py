from typing import Sequence, Union, Dict, Tuple
import tensorflow as tf

Inputs = Union[tf.Tensor, Sequence[tf.Tensor], Dict[str, tf.Tensor]]
Outputs = Union[tf.Tensor, Sequence[tf.Tensor], Dict[str, tf.Tensor]]
Losses = Union[tf.Tensor, Sequence[tf.Tensor]]
Gradients = Sequence[Tuple[tf.Tensor, tf.Variable]]
