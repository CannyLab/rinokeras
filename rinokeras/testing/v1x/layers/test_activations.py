import tensorflow as tf
import numpy as np
import json
import os
import warnings

from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import check_regression

def test_gated_tanh():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.layers.activations import GatedTanh
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
                     output, __file__, 'regression_outputs/test_activation_outputs.json', debug=_RK_REBUILD_REGRESSION)
