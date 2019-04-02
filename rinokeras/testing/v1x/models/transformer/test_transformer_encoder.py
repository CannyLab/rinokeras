"""
Testing for Transformer encoder layers
"""
import tensorflow as tf
import tempfile

from rinokeras.core.v1x.utils import convert_to_attention_mask, convert_sequence_length_to_sequence_mask
from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import reset_session, random_tensor, run_simple_session_save_weights,\
        assert_not_none, assert_expected_shapes, load_restore_test, check_regression, \
        from_config_test, random_mask_tensor

def test_transformer_encoder_block():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerEncoderBlock
    layer = TransformerEncoderBlock(n_heads=4, filter_size=128, hidden_size=64)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,32,64))

    # Get the output of the layer
    value = layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(16,32,64)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_encoder_block_output',
                     output, __file__, 'regression_outputs/test_transformer_encoder_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Do a config test
    from_config_test(TransformerEncoderBlock, layer)

def test_transformer_encoder_block_masking():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerEncoderBlock
    layer = TransformerEncoderBlock(n_heads=4, filter_size=128, hidden_size=64)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,32,64))
    input_mask, _ = random_mask_tensor(16, 32)
    input_mask = convert_to_attention_mask(input_tensor, input_mask)

    # Get the output of the layer
    value = layer(input_tensor, mask=input_mask)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(16,32,64)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_encoder_block_masking_output',
                     output, __file__, 'regression_outputs/test_transformer_encoder_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Do a config test
    from_config_test(TransformerEncoderBlock, layer)


def test_transformer_encoder():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerEncoder
    layer = TransformerEncoder(
        embedding_layer=tf.keras.layers.Dense(64),
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_filter=128
    )
    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,32,128))

    # Get the output of the layer
    value = layer(input_tensor)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(16,32,64)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_encoder_output',
                     output, __file__, 'regression_outputs/test_transformer_encoder_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Do a config test
    from_config_test(TransformerEncoder, layer)

def test_transformer_encoder_masking():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerEncoder
    layer = TransformerEncoder(
        embedding_layer=tf.keras.layers.Dense(64),
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_filter=128
    )
    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,32,128))
    input_mask, _ = random_mask_tensor(16, 32)
    input_mask = convert_to_attention_mask(input_tensor, input_mask)
    
    # Get the output of the layer
    value = layer(input_tensor, mask=input_mask)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(16,32,64)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_encoder_output_masking',
                     output, __file__, 'regression_outputs/test_transformer_encoder_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Do a config test
    from_config_test(TransformerEncoder, layer)

def test_transformer_encoder_masking_with_conv():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerEncoder
    layer = TransformerEncoder(
        embedding_layer=tf.keras.layers.Dense(64),
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_filter=128
    )
    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16,32,128))
    input_mask, _ = random_mask_tensor(16, 32)
    input_mask = convert_to_attention_mask(input_tensor, input_mask)
    conv_mask, _ = random_mask_tensor(16, 32)
    conv_mask = convert_sequence_length_to_sequence_mask(input_tensor, conv_mask)
    
    # Get the output of the layer
    value = layer(input_tensor, mask=(input_mask, conv_mask))

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(16,32,64)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_encoder_output_masking_with_conv',
                     output, __file__, 'regression_outputs/test_transformer_encoder_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Do a config test
    from_config_test(TransformerEncoder, layer)

