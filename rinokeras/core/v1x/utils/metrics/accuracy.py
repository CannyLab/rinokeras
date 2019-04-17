import tensorflow as tf
import tensorflow.keras.backend as K


def accuracy(labels, predictions, weights=1.0, dtype=None):
    if dtype is None:
        dtype = K.floatx()
    is_correct = tf.equal(labels, predictions)
    is_correct = K.cast(is_correct, dtype)

    weights = tf.ones_like(is_correct) * weights
    return tf.reduce_sum(is_correct * weights) / (tf.reduce_sum(weights) + 1e-10)
