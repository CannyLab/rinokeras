
"""
Testing for Transformer decoder layers
"""
import tensorflow as tf
import tempfile

from rinokeras.core.v1x.utils import convert_to_attention_mask
from rinokeras.testing.utils import reset_session, random_tensor, run_simple_session_save_weights,\
        assert_not_none, assert_expected_shapes, load_restore_test, check_regression, \
        from_config_test, random_mask_tensor


def test_transformer_decoder_block_no_mask():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerDecoderBlock
    layer = TransformerDecoderBlock(n_heads=4, filter_size=128, hidden_size=64)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    source_tensor, _ = random_tensor((2,32,64))
    target_tensor, _ = random_tensor((2,32,64))

    # Get the output of the layer
    _, value = layer((source_tensor, target_tensor))

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,32,64)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_decoder_block_no_mask_output',
                     output, __file__, 'regression_outputs/test_transformer_decoder_outputs.json')

    # Do a config test
    from_config_test(TransformerDecoderBlock, layer)

def test_transformer_decoder_block():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerDecoderBlock
    layer = TransformerDecoderBlock(n_heads=4, filter_size=128, hidden_size=64)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    source_tensor, _ = random_tensor((2,32,64))
    target_tensor, _ = random_tensor((2,32,64))
    
    # Get random masking values
    source_mask, _ = random_mask_tensor(2, 32)
    target_mask, _ = random_mask_tensor(2, 32)
    source_mask = convert_to_attention_mask(source_tensor, source_mask)
    target_mask = convert_to_attention_mask(target_tensor, target_mask)

    # Get the output of the layer
    _, value = layer((source_tensor, target_tensor), mask=(source_mask,target_mask))

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,32,64)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_decoder_block_output',
                     output, __file__, 'regression_outputs/test_transformer_decoder_outputs.json')

    # Do a config test
    from_config_test(TransformerDecoderBlock, layer)

def test_transformer_decoder_no_mask():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerDecoder, TransformerInputEmbedding
    tie = TransformerInputEmbedding(128, False)
    layer = TransformerDecoder(embedding_layer=tie,
                               output_layer=tf.keras.layers.Dense(128),
                               n_layers=2,
                               n_heads=4,
                               d_model=128,
                               d_filter=32)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    source_tensor, _ = random_tensor((2,32,128))
    target_tensor, _ = random_tensor((2,32,64))
    
    # Get random masking values
    source_mask, _ = random_mask_tensor(2, 32)
    target_mask, _ = random_mask_tensor(2, 32)
    source_mask = convert_to_attention_mask(source_tensor, source_mask)
    target_mask = convert_to_attention_mask(target_tensor, target_mask)

    # Get the output of the layer
    value = layer((source_tensor, target_tensor), mask=None)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,32,128)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_decoder_no_mask_output',
                     output, __file__, 'regression_outputs/test_transformer_decoder_outputs.json')

    # Do a config test
    from_config_test(TransformerDecoder, layer)

def test_transformer_decoder():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerDecoder, TransformerInputEmbedding
    tie = TransformerInputEmbedding(128, False)
    layer = TransformerDecoder(embedding_layer=tie,
                               output_layer=tf.keras.layers.Dense(128),
                               n_layers=2,
                               n_heads=4,
                               d_model=128,
                               d_filter=32)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    source_tensor, _ = random_tensor((2,32,128))
    target_tensor, _ = random_tensor((2,32,64))
    
    # Get random masking values
    source_mask, _ = random_mask_tensor(2, 32)
    target_mask, _ = random_mask_tensor(2, 32)
    source_mask = convert_to_attention_mask(source_tensor, source_mask)
    target_mask = convert_to_attention_mask(target_tensor, target_mask)

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
        assert_expected_shapes(output, [(2,32,128)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_decoder_output',
                     output, __file__, 'regression_outputs/test_transformer_decoder_outputs.json')

    # Do a config test
    from_config_test(TransformerDecoder, layer)

def test_transformer_decoder_fast_decode():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerDecoder, TransformerInputEmbedding
    tie = TransformerInputEmbedding(128, False)
    layer = TransformerDecoder(embedding_layer=tie,
                               output_layer=tf.keras.layers.Dense(128),
                               n_layers=2,
                               n_heads=4,
                               d_model=128,
                               d_filter=32)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    source_tensor, _ = random_tensor((2,32,128))

    # Get the output of the layer
    value = layer.fast_decode(source_tensor, 20, output_size=128)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,20,128)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_decoder_fast_decode',
                     output, __file__, 'regression_outputs/test_transformer_decoder_outputs.json')

    # Do a config test
    from_config_test(TransformerDecoder, layer)

def test_transformer_decoder_fast_decode_discrete():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerDecoder, TransformerInputEmbedding
    tie = TransformerInputEmbedding(128, discrete=True, n_symbols=256)
    layer = TransformerDecoder(embedding_layer=tie,
                               output_layer=tf.keras.layers.Dense(256),
                               n_layers=2,
                               n_heads=4,
                               d_model=128,
                               d_filter=32)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    source_tensor, _ = random_tensor((2,32,128))

    # Get the output of the layer
    value = layer.fast_decode(source_tensor, 20, output_dtype=tf.int32)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,20)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_decoder_fast_decode_discrete',
                     output, __file__, 'regression_outputs/test_transformer_decoder_outputs.json')

    # Do a config test
    from_config_test(TransformerDecoder, layer)

def test_transformer_decoder_fast_beam_decode_discrete():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.models.transformer import TransformerDecoder, TransformerInputEmbedding
    tie = TransformerInputEmbedding(128, discrete=True, n_symbols=256)
    layer = TransformerDecoder(embedding_layer=tie,
                               output_layer=tf.keras.layers.Dense(256),
                               n_layers=2,
                               n_heads=4,
                               d_model=128,
                               d_filter=32)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    source_tensor, _ = random_tensor((2,32,128))

    # Get the output of the layer
    value, scores = layer.fast_beam_decode(source_tensor, 20, batch_size=2, n_beams=4)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value, scores],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        assert_not_none(output)
        assert_expected_shapes(output, [(2,4, 20), (2,4)])

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('transformer_decoder_fast_beam_decode',
                     output, __file__, 'regression_outputs/test_transformer_decoder_outputs.json')

    # Do a config test
    from_config_test(TransformerDecoder, layer)