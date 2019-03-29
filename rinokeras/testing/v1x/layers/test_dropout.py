import numpy as np
import tensorflow as tf
import warnings
import tempfile

from rinokeras.testing.utils import *

def test_layer_dropout():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.dropout import LayerDropout
    
    dense_layer = tf.keras.layers.Dense(128)
    layer_drop_1 = LayerDropout(rate=5.0)

    # Make sure that the layer is not None
    assert layer_drop_1 is not None

    # Encoded values
    input_tensor, input_values = random_tensor((16, 128))

    # Get the output of the layer
    training = tf.placeholder(tf.bool, [])
    value = layer_drop_1(dense_layer(input_tensor), input_tensor, training=training)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output_drop = run_simple_session(inputs=[value],
                                         feed={training: True})

        output_no_drop = run_simple_session_save_weights(inputs=[value],
                                                 feed={training: False},
                                                 weights=[layer_drop_1],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output_drop[0] is not None
        assert output_no_drop[0] is not None

        # Make sure the output shape is correct
        assert output_drop[0].shape == (16,128)
        assert output_no_drop[0].shape == (16, 128)

        # Make sure the output values are correct (If Possible)
        assert not np.isclose(output_drop[0], output_no_drop[0]).all()
        assert np.isclose(output_drop[0], input_values).all()

        # Check loading and restoring
        load_restore_test(output=output_no_drop,
                          inputs=[value],
                          feed={training: False},
                          weights=[layer_drop_1],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('layer_dropout_nd_expected_output',
                     output_no_drop, __file__, 'regression_outputs/test_dropout_outputs.json')
    check_regression('layer_dropout_expected_output',
                     output_drop, __file__, 'regression_outputs/test_dropout_outputs.json')
