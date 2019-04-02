"""
Testing for Stack layers
"""
import numpy as np
import tensorflow as tf
import warnings
import tempfile

from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import *


def test_stack():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.stack import Stack
    layer = Stack([
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
    ])

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 128))

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

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('stack_expected_output',
                     output, __file__, 'regression_outputs/test_stack_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_layer_dropout_stack():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.stack import LayerDropoutStack
    layer = LayerDropoutStack([
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
    ], layer_dropout=0.5)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 128))

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

        #TODO: Make sure that the layer dropout stack is working in test/train mode

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('layer_dropout_stack_expected_output',
                     output, __file__, 'regression_outputs/test_stack_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_conv2d_stack():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.stack import Conv2DStack
    layer = Conv2DStack([8, 8, 8], [4, 4, 4], [1, 1, 1])

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 8, 8, 3))

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
        assert output[0].shape == (16, 512)

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('conv2d_stack_expected_output',
                     output, __file__, 'regression_outputs/test_stack_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_deconv2d_stack():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.stack import Deconv2DStack
    layer = Deconv2DStack([8, 8, 8], [4, 4, 4], [
                          1, 1, 1], flatten_output=False)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 64, 64, 3))

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
        assert output[0].shape == (16, 64, 64, 8)

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('deconv2d_stack_expected_output',
                     output, __file__, 'regression_outputs/test_stack_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_dense_stack():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.stack import DenseStack
    layer = DenseStack([128, 128, 128])

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 32))

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

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('dense_stack_expected_output',
                     output, __file__, 'regression_outputs/test_stack_outputs.json', debug=_RK_REBUILD_REGRESSION)
