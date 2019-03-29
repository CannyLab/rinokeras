import numpy as np
import tensorflow as tf
import warnings
import tempfile

from rinokeras.testing.utils import *

def test_dense_transpose():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.inversion import DenseTranspose

    dense_layer = tf.keras.layers.Dense(64, use_bias=True)
    layer = DenseTranspose(dense_layer)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, input_values = random_tensor((16, 128))

    # Get the output of the layer
    value = layer(dense_layer(input_tensor))

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
    check_regression('dense_transpose_layer_expected_output',
                     output, __file__, 'regression_outputs/test_inversion_outputs.json')

def test_embedding_transpose():

    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.inversion import EmbeddingTranspose

    embedding_layer = tf.keras.layers.Embedding(32, 12)
    layer = EmbeddingTranspose(embedding_layer)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor = tf.expand_dims(tf.convert_to_tensor(np.arange(32)), axis=0)

    # Get the output of the layer
    value = layer(embedding_layer(input_tensor))

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
        assert output[0].shape == (1, 32, 32)
        
        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('embedding_transpose_layer_expected_output',
                     output, __file__, 'regression_outputs/test_inversion_outputs.json')

def test_invertible_dense():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.inversion import InvertibleDense

    layer = InvertibleDense(8)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, input_values = random_tensor((16, 8, 8))

    # Get the output of the layer
    value, log_det_w = layer(input_tensor)
    rev_value = layer(value, reverse=True)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value, log_det_w, rev_value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None
        assert output[1] is not None
        assert output[2] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 8, 8)
        assert output[1].shape == ()
        assert output[2].shape == (16, 8, 8)
        
        # Make sure the output values are correct (If Possible)
        assert np.isclose(output[2], input_values, rtol=.01, atol=.01).all()

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value, log_det_w, rev_value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('invertible_dense_layer_expected_output',
                     output, __file__, 'regression_outputs/test_inversion_outputs.json', tol=1e-1)
