"""
Testing for Transformer FF layers
"""
import tempfile

from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import reset_session, random_tensor, run_simple_session_save_weights,\
        assert_not_none, assert_expected_shapes, load_restore_test, check_regression, \
        from_config_test

def test_transformer_input_embedding_non_discrete():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerInputEmbedding
    layer = TransformerInputEmbedding(embed_size=128, discrete=False)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,32,64))

    # Get the output of the layerembedding_layer
    value = layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(16,32,128)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_input_embedding_non_discrete_expected_output',
                     output, __file__, 'regression_outputs/test_transformer_embedding_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Do a config test
    from_config_test(TransformerInputEmbedding, layer)
