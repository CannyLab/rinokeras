import os
import tensorflow as tf
import numpy as np

from rinokeras.models.qanet import QANet


def test_qanet():
    word_embed_matrix = np.random.random((1000, 300))
    char_embed_matrix = np.random.random((26, 300))
    qanet = QANet(word_embed_matrix, char_embed_matrix,
                  dropout=0.1)

    context = np.random.randint(0, 1000, (32, 100), dtype=np.int32)
    query = np.random.randint(0, 1000, (32, 10), dtype=np.int32)
    context_chars = np.random.randint(0, 1000, (32, 100, 16), dtype=np.int32)
    query_chars = np.random.randint(0, 1000, (32, 10, 16), dtype=np.int32)
    answer_index_1 = np.random.randint(0, 100, (32,), dtype=np.int32)
    answer_index_2 = np.random.randint(0, 100, (32,), dtype=np.int32)

    inputs = (context, query, context_chars, query_chars,
              answer_index_1, answer_index_2)
    inputs = tuple(tf.constant(el) for el in inputs)
    inputs = inputs + (None,)
    output = qanet(inputs)

    output = output.numpy()
    assert output.shape == (32, 100, qanet.d_model), 'Output shape does not match'
