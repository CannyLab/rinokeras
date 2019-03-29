"""

Invertible layers, such as the InvertibleDense layer and the DenseTranspose
layers.

"""
import numpy as np
import scipy

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.keras import Model  # pylint: disable=F0401
from tensorflow.keras.layers import Layer, Dense, Embedding   # pylint: disable=F0401
import tensorflow.keras.backend as K  # pylint: disable=F0401



class DenseTranspose(Layer):
    """Multiply by the transpose of a dense layer
    """
    def __init__(self, other_layer):
        super(DenseTranspose, self).__init__()
        self.other_layer = other_layer

    def call(self, x):
        return K.dot(x - K.stop_gradient(self.other_layer.bias), K.transpose(K.stop_gradient(self.other_layer.kernel)))


class EmbeddingTranspose(Model):
    """Multiply by the transpose of an embedding layer
    """
    def __init__(self, embedding_layer: Embedding, *args, **kwargs) -> None:
        super(EmbeddingTranspose, self).__init__(*args, **kwargs)
        self.embedding = embedding_layer

    def call(self, inputs):
        embed_mat = self.embedding.weights[0]
        return K.dot(inputs, K.stop_gradient(K.transpose(embed_mat)))


class InvertibleDense(Dense):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_bias=False, kernel_initializer=None, activation=None, **kwargs)

    def build(self, input_shape):
        assert input_shape[-1] == self.units, \
            'Cannot create invertible layer when mapping from {} to {} dimensions'.format(
                input_shape[-1].value, self.units)
        H = np.random.randn(input_shape[-1].value, self.units)
        Q, _ = scipy.linalg.qr(H)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        initializer = tf.keras.initializers.Constant(Q)

        self.kernel_initializer = initializer
        super().build(input_shape)
        self.kernel_inverse = tf.linalg.inv(self.kernel)

    def call(self, inputs, reverse=False):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)

        kernel = self.kernel if not reverse else self.kernel_inverse

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, kernel)

        if reverse:
            return outputs

        batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
        sequence_length = tf.cast(tf.shape(inputs)[1], tf.float32)
        # tf.logdet only works on Hermitian positive def matrices, maybe try tf.slogdet?
        log_det_W = batch_size * sequence_length * tf.log(tf.linalg.det(self.kernel))
        return outputs, log_det_W
