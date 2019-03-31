"""
Utilities for testing tensors
"""
import tensorflow as tf
import numpy as np

def random_tensor(shape, dtype=np.float32):
    rv = np.random.sample(shape).astype(dtype)
    return tf.convert_to_tensor(rv), rv

def random_mask_tensor(batch_size, seqlen, dtype=np.int32):
    rv = np.random.randint(0,seqlen,size=batch_size).astype(dtype)
    return tf.convert_to_tensor(rv), rv

def random_sequence_tensor(batch_size, seqlen, n_symbols,dtype=np.int32):
    rv = np.random.randint(0,n_symbols,size=(batch_size, seqlen)).astype(dtype)
    return tf.convert_to_tensor(rv), rv