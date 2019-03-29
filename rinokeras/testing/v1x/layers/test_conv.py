import numpy as np
import warnings
import tempfile

from rinokeras.testing.utils import check_regression, load_restore_test, \
    random_tensor, reset_session, run_simple_session_save_weights

def test_normed_conv_stack_1d():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.conv import NormedConvStack
    ncs_layer = NormedConvStack(dimension=1, filters=12, kernel_size=4)

    # Make sure that the layer is not None
    assert ncs_layer is not None

    # Encoded values
    input_tensor, input_values = random_tensor((16, 8, 32))

    # Get the output of the layer
    value = ncs_layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[ncs_layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 8, 12)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[ncs_layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('normed_conv_stack_1d_expected_output',
                     output, __file__, 'regression_outputs/test_conv_outputs.json')

def test_normed_conv_stack_2d():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.conv import NormedConvStack
    ncs_layer = NormedConvStack(dimension=2, filters=12, kernel_size=4)

    # Make sure that the layer is not None
    assert ncs_layer is not None

    # Encoded values
    input_tensor, input_values = random_tensor((16, 8, 32, 32))

    # Get the output of the layer
    value = ncs_layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[ncs_layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 8, 32, 12)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[ncs_layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('normed_conv_stack_2d_expected_output',
                     output, __file__, 'regression_outputs/test_conv_outputs.json')

def test_normed_conv_stack_3d():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.conv import NormedConvStack
    ncs_layer = NormedConvStack(dimension=3, filters=12, kernel_size=4)

    # Make sure that the layer is not None
    assert ncs_layer is not None

    # Encoded values
    input_tensor, input_values = random_tensor((16, 8, 32, 32, 32))

    # Get the output of the layer
    value = ncs_layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[ncs_layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 8, 32, 32, 12)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[ncs_layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('normed_conv_stack_3d_expected_output',
                     output, __file__, 'regression_outputs/test_conv_outputs.json')

def test_residual_block():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.conv import ResidualBlock
    rb_layer = ResidualBlock(dimension=2, filters=32, kernel_size=4, n_layers=2)

    # Make sure that the layer is not None
    assert rb_layer is not None

    # Encoded values
    input_tensor, input_values = random_tensor((16, 8, 32, 32))

    # Get the output of the layer
    value = rb_layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[rb_layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 8, 32, 32)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[rb_layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('residual_block_2d_expected_output',
                     output, __file__, 'regression_outputs/test_conv_outputs.json')
