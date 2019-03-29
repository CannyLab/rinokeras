"""
Testing for masking
"""
import numpy as np
import tensorflow as tf
import warnings
import tempfile

from rinokeras.testing.utils import *

def test_bert_random_replace_mask_discrete():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.masking import BERTRandomReplaceMask
    layer = BERTRandomReplaceMask(percentage=0.5, mask_token=1, n_symbols=2)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor = tf.convert_to_tensor(np.zeros((16,8,256)), dtype=np.int32)

    # Get the output of the layer
    value, bm = layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value, bm],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None
        assert output[1] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 8, 256)
        assert output[1].shape == (16, 8, 256)

        # Make sure the output values are correct (If Possible)
        assert np.sum(output[0]) > 0
        assert np.sum(output[0]) < 8*256*16

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value, bm],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('random_replace_mask_discrete_expected_output',
                     output, __file__, 'regression_outputs/test_masking_outputs.json')

def test_bert_random_replace_mask_floating():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.masking import BERTRandomReplaceMask
    layer = BERTRandomReplaceMask(percentage=0.5, mask_token=1, n_symbols=2)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor = tf.convert_to_tensor(np.zeros((16,8,256)), dtype=np.float32)

    # Get the output of the layer
    value, bm = layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value, bm],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None
        assert output[1] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16, 8, 256)
        assert output[1].shape == (16, 8, 1)

        # Make sure the output values are correct (If Possible)
        assert np.sum(output[0]) > 0
        assert np.sum(output[0]) < 8*256*16

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value, bm],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('random_replace_mask_float_expected_output',
                     output, __file__, 'regression_outputs/test_masking_outputs.json')