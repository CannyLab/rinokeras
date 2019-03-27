import tensorflow as tf
import numpy as np
import json
import os
import warnings
import tempfile
import pickle

def get_local_file(fpath):
    return '/'+os.path.join(*__file__.split(os.sep)[:-1], fpath)

def check_regression(regression_key, output, fname, debug=False):
    try:
        with open(get_local_file(fname), 'r') as json_file:
            jf = json.loads(json_file.read())
    except FileNotFoundError:
        warnings.warn('{} not found. Creating it.'.format(fname))
        jf = {}
    if not debug and regression_key in jf:
        with open(get_local_file(fname), 'r') as json_file:
            jf = json.loads(json_file.read())
            expected_output = [np.array(v) for v in jf[regression_key]]
    else:
        if isinstance(output, (list, tuple)):
            jf[regression_key] = [i.tolist() for i in output]
            expected_output = output
        else:
            jf[regression_key] = [output.tolist()]
            expected_output = [output]
        with open(get_local_file(fname), 'w') as json_file:
            json.dump(jf, json_file)
        warnings.warn('Regression test not found for {} in {}: Building this\
             now.'.format(regression_key, fname))

    # Now do assertions
    if not isinstance(output, (list, tuple)):
        output = [output]
    for x, y in zip(output, expected_output):
        assert np.isclose(x, y).all()


def test_random_gauss_noise():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.v1x.common.layers.autoregressive import RandomGaussNoise
    gaussian_noise_layer = RandomGaussNoise()
    assert gaussian_noise_layer is not None

    # Encoded values
    input_values = np.random.sample((16, 256))

    # Get some sample input tensors
    input_tensor = tf.constant(input_values)

    value = gaussian_noise_layer(input_tensor)
    logstd = gaussian_noise_layer.logstd
    std = gaussian_noise_layer.std

    # Create a named temporary file for save/restore testing
    output_file = tempfile.NamedTemporaryFile()

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run([value, logstd, std])

        # Check weights
        weights = gaussian_noise_layer.get_weights()
        pickle.dump(weights, output_file)

    assert output[0] is not None  # Make sure the value is not null
    assert output[1] is not None  # Make sure the value is not null
    assert output[2] is not None  # Make sure the value is not null
    assert output[0].shape == (16, 256)  # Make sure the output shape is correct
    assert output[1].shape == (256,)  # Make sure the output shape is correct
    assert output[2].shape == (256,)  # Make sure the output shape is correct
    assert np.isclose(output[1], np.zeros_like(output[1])).all()  # Make sure the output value is correct
    assert np.isclose(output[2], np.ones_like(output[2])).all()  # Make sure the output value is correct
    assert not np.isclose(output[0], input_values).all() # Make sure some noise was added

    # Load/Restore test
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output_file.seek(0)
        gaussian_noise_layer.set_weights(pickle.load(output_file))        
        restored_output = sess.run([value, logstd, std])
    assert np.isclose(restored_output[0], output[0]).all()
    output_file.close()

    # Do regression testing
    check_regression('random_gauss_noise_expected_output',
                     output, 'regression_outputs/test_autoregressive_outputs.json')

def test_coupling_layer():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.v1x.common.layers.autoregressive import CouplingLayer
    coupling_layer = CouplingLayer(n_units=128, layer=tf.keras.layers.Dense(128))
    assert coupling_layer is not None

    # Encoded values
    input_values_a = np.random.sample((16, 128))
    input_values_b = np.random.sample((16, 128))

    # Get some sample input tensors
    input_tensor_a = tf.constant(input_values_a)
    input_tensor_b = tf.constant(input_values_b)

    value, log_s = coupling_layer((input_tensor_a, input_tensor_b))
    reverse_value = coupling_layer((input_tensor_a, input_tensor_b), reverse=True)

    # Create a named temporary file for save/restore testing
    output_file = tempfile.NamedTemporaryFile()

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run([value, log_s, reverse_value])

        # Check weights
        weights = coupling_layer.get_weights()
        pickle.dump(weights, output_file)

    assert output[0] is not None  # Make sure the value is not null
    assert output[1] is not None  # Make sure the value is not null
    assert output[2] is not None  # Make sure the value is not null
    assert output[0].shape == (16,128)  # Make sure the output shape is correct
    assert output[1].shape == (16,128)  # Make sure the output shape is correct
    assert output[2].shape == (16,128)  # Make sure the output shape is correct

    # Load/Restore test
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output_file.seek(0)
        coupling_layer.set_weights(pickle.load(output_file))        
        restored_output = sess.run([value, log_s, reverse_value])
    assert np.isclose(restored_output[0], output[0]).all()
    assert np.isclose(restored_output[1], output[1]).all()
    assert np.isclose(restored_output[2], output[2]).all()
    output_file.close()

    # Do regression testing
    check_regression('coupling_layer_expected_output',
                     output, 'regression_outputs/test_autoregressive_outputs.json')