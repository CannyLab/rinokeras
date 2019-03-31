"""
Utilities for testing load/restore of a layer
"""
import tensorflow as tf
import pickle
import json
import copy

def load_restore_test(output, inputs, feed, weights, weights_file):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Load the weights
        weights_file.seek(0)
        weights_data = pickle.load(weights_file)
        assert len(weights_data) == len(weights), "Length of saved and restored \
            weights should be equivalent."
        for idx, w in enumerate(weights):
            w.set_weights(weights_data[idx])

        output = sess.run(inputs,feed_dict=feed)
    return output


def from_config_test(__cls, __obj):
    config = __obj.get_config()
    if config is None:
        raise AssertionError('Cannot properly serialize the layer/model')
    new_obj = __cls.from_config(config)
    if new_obj is None:
        raise AssertionError('Cannot properly deserialize the layer/model')
    if json.dumps(new_obj.get_config()) != json.dumps(__obj.get_config()):
        raise AssertionError('Construction from configutation failed')

