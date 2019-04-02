import tensorflow as tf

def accuracy(labels, predictions, weights=None, dtype=tf.float32):
    is_correct = tf.equal(labels, predictions)
    is_correct = tf.cast(is_correct, dtype)
    if not weights:
        return tf.reduce_mean(is_correct)
    
    weights = tf.ones_like(is_correct) * weights
    return tf.reduce_sum(is_correct * weights) / tf.reduce_sum(weights)
