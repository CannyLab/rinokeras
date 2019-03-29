"""
Testing for normalization layers
"""
import numpy as np
import tensorflow as tf
import warnings
import tempfile

from rinokeras.testing.utils import *

def test_layer_norm():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.normalization import LayerNorm
    layer = LayerNorm()

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,8,32))

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
        assert output[0].shape == (16, 8, 32)

        # Make sure the output values are correct (If Possible)
        assert np.isclose(np.mean(output[0], axis=-1), np.zeros((16, 8)), rtol=1e-2, atol=1e-2).all()
        assert np.isclose(np.var(output[0], axis=-1), np.ones((16, 8)), rtol=1e-2, atol=1e-2).all()

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('layer_norm_expected_output',
                     output, __file__, 'regression_outputs/test_normalization_outputs.json')

def test_weight_norm_dense():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.normalization import WeightNormDense
    layer = WeightNormDense(128)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 8, 32))

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
        assert output[0].shape == (16, 8, 128)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('weight_norm_dense_expected_output',
                     output, __file__, 'regression_outputs/test_normalization_outputs.json')