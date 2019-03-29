"""
Utilities for testing load/restore of a layer
"""
import tensorflow as tf
import pickle

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

