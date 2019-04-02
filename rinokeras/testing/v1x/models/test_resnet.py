"""
Testing for ResNet models
"""
import tempfile

import tensorflow as tf
import numpy as np

from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import reset_session, random_tensor, run_simple_session_save_weights,\
        assert_not_none, assert_expected_shapes, load_restore_test, check_regression, \
        from_config_test, random_sequence_tensor

def test_residual_block():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.resnet.resnet import ResidualBlock
    layer = ResidualBlock()
    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    inputs_tensor , _ = random_tensor((2,8,8,64))

    # Get the output of the layer
    value = layer(inputs_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,8,8,64)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('residual_block_expected_output',
                     output, __file__, 'regression_outputs/test_resnet_outputs.json', debug=_RK_REBUILD_REGRESSION)

def test_resnext50_base():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.resnet.resnet import ResNeXt50
    layer = ResNeXt50()
    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    inputs_tensor , _ = random_tensor((2,8,8,3))

    # Get the output of the layer
    value = layer(inputs_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,2048)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('resnext50_base_expected_output',
                     output, __file__, 'regression_outputs/test_resnet_outputs.json', debug=_RK_REBUILD_REGRESSION)