import tensorflow as tf
import numpy as np
import json
import os
import warnings

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


def test_gated_tanh():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.v1x.common.layers.activations import GatedTanh
    gth_layer = GatedTanh(n_units=128)
    assert gth_layer is not None

    # Encoded values
    input_values = np.random.sample((16, 256))

    # Get some sample input tensors
    input_tensor = tf.constant(input_values)
    value = gth_layer(input_tensor)

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output is not None  # Make sure the value is correct
    assert output.shape == (16, 128)  # Make sure the output shape is correct

    # Do regression testing
    check_regression('gated_tanh_expected_output',
                     output, 'regression_outputs/test_activation_outputs.json')
