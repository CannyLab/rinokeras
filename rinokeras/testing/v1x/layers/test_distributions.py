import numpy as np
import tensorflow as tf
import warnings
import tempfile


from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import check_regression, load_restore_test, \
    random_tensor, reset_session, run_simple_session_save_weights

def test_CategoricalPd():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.distributions import CategoricalPd
    layer = CategoricalPd()

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 128))

    # Get the output of the layer
    value = layer(input_tensor)
    logp_actions = layer.logp_actions(tf.reshape(tf.range(0,16), (-1, 1)))
    entropy = layer.entropy()

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value, logp_actions, entropy],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None
        assert output[1] is not None
        assert output[2] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16,)
        assert output[1].shape == (16,16)
        assert output[2].shape == (16,)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value, logp_actions, entropy],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('CategoricalPd_expected_output',
                     output, __file__, 'regression_outputs/test_distribution_outputs.json', debug=_RK_REBUILD_REGRESSION)

def test_DiagGaussianPd():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.distributions import DiagGaussianPd
    layer = DiagGaussianPd(action_shape=(4,4))

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 4, 4))
    actions_tensor, _ = random_tensor((16, 4, 4))

    # Get the output of the layer
    value = layer(input_tensor)
    logp_actions = layer.logp_actions(actions_tensor)
    entropy = layer.entropy()

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value, logp_actions, entropy],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None
        assert output[1] is not None
        assert output[2] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 4, 4)
        assert output[1].shape == (16,)
        assert output[2].shape == ()

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value, logp_actions, entropy],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('DiagGaussianPd_expected_output',
                     output, __file__, 'regression_outputs/test_distribution_outputs.json', debug=_RK_REBUILD_REGRESSION)