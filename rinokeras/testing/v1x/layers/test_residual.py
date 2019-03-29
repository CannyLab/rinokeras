"""
Testing for position embedding layers
"""
import numpy as np
import tensorflow as tf
import warnings
import tempfile

from rinokeras.testing.utils import *

def test_residual():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.residual import Residual
    dense_layer = tf.keras.layers.Dense(128)
    layer = Residual(dense_layer)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,128))

    # Get the output of the layer
    value = layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 128)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('residual_expected_output',
                     output, __file__, 'regression_outputs/test_residual_outputs.json')

def test_highway():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.residual import Highway
    dense_layer = tf.keras.layers.Dense(128)
    layer = Highway(dense_layer)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,128))

    # Get the output of the layer
    value = layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 128)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('highway_expected_output',
                     output, __file__, 'regression_outputs/test_residual_outputs.json')