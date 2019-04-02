"""
Testing for the transformer
"""
import tensorflow as tf
import tempfile

from rinokeras.core.v1x.utils import convert_to_attention_mask
from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import reset_session, random_tensor, run_simple_session_save_weights,\
        assert_not_none, assert_expected_shapes, load_restore_test, check_regression, \
        from_config_test, random_mask_tensor, random_sequence_tensor

def test_transformer_base():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import Transformer
    layer = Transformer(
        discrete=True,
        n_symbols_in=128,
        n_symbols_out=128,
        n_layers=6,
        n_heads=8,
        d_model=32,
        d_filter=128)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    source_tensor, _ = random_sequence_tensor(2, 32, 128)
    target_tensor, _ = random_sequence_tensor(2, 32, 128)
    source_mask, _ = random_mask_tensor(2,32)
    target_mask, _ = random_mask_tensor(2,32)

    # Get the output of the layer
    value = layer((source_tensor, target_tensor), mask=(source_mask, target_mask))

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output,[(2,32, 128)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_base',
                     output, __file__, 'regression_outputs/test_transformer_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Do a config test
    from_config_test(Transformer, layer)
