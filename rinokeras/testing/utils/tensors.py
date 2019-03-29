"""
Utilities for testing tensors
"""
import tensorflow as tf
import numpy as np

def random_tensor(shape, dtype=np.float32):
    rv = np.random.sample(shape).astype(dtype)
    return tf.convert_to_tensor(rv), rv
