"""
Session utils for testing
"""
import tensorflow as tf
import numpy as np
import random as rn
import pickle

def run_simple_session_save_weights(inputs, feed, weights, weights_file):
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        tf.keras.backend.set_session(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(inputs,feed_dict=feed)
        # Save the weights to the temporary file
        saved_weights = []
        for w in weights:
            saved_weights.append(w.get_weights())
        pickle.dump(saved_weights, weights_file)
    return output

def run_simple_session(inputs, feed):
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        tf.keras.backend.set_session(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(inputs,feed_dict=feed)
    return output

def reset_session():
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    np.random.seed(256)
    rn.seed(256)
    tf.set_random_seed(256)
    
    
