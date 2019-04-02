"""
Testing for QANet Embedding layers
"""
import tempfile

import tensorflow as tf
import numpy as np

from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import reset_session, random_tensor, run_simple_session_save_weights,\
        assert_not_none, assert_expected_shapes, load_restore_test, check_regression, \
        from_config_test, random_sequence_tensor

def test_qanet_input_embedding():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.qanet import QANetInputEmbedding
    layer = QANetInputEmbedding(
        d_model=32,
        word_embed_initializer=np.random.sample((128, 32)),
        char_embed_initializer=np.random.sample((128, 32)))

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_sequence_tensor(2, 8, 32)
    input_char_tensor = tf.convert_to_tensor(np.random.randint(0, 32, (2,8,16)))

    # Get the output of the layer
    value = layer((input_tensor, input_char_tensor))

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,8,32)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('qanet_input_embedding_expected_output',
                     output, __file__, 'regression_outputs/test_qanet_embedding_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Do a config test
    from_config_test(QANetInputEmbedding, layer)