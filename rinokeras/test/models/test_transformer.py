
import numpy as np
import tensorflow as tf
import sys

from rinokeras.models.transformer import Transformer


def test_transformer():
    transformer = Transformer(True, 1000, 1000, dropout=0.1,
                              embedding_initializer=np.random.random((1000, 300)))

    source = np.random.randint(0, 1000, (32, 100), dtype=np.int32)
    target = np.random.randint(0, 1000, (32, 10), dtype=np.int32)

    source = tf.constant(source)
    target = tf.constant(target)

    output = transformer(source, target)

    output = output.numpy()
    assert output.shape == (32, 10, 1000), 'Output shape does not match'
