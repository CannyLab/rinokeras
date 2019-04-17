"""
Testing for position embedding layers
"""
import numpy as np
import tensorflow as tf
import warnings
import tempfile

from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import *

def test_position_embedding_vanilla():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding
    layer = PositionEmbedding()

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,8,32))

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
        assert output[0].shape == (2, 8, 32)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_vanilla_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_position_embedding_concat():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding
    layer = PositionEmbedding(concat=True)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,8,32))

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
        assert output[0].shape == (2, 8, 64)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_concat_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_position_embedding_concat_reproject():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding
    layer = PositionEmbedding(concat=True, reproject_embedding=True)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,8,32))

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
        assert output[0].shape == (2, 8, 32)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_concat_reproject_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_position_embedding_2d_vanilla():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding2D
    layer = PositionEmbedding2D()

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,8,8,32))

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
        assert output[0].shape == (2, 8, 8,32)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_2d_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_position_embedding_2d_concat():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding2D
    layer = PositionEmbedding2D(concat=True)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,32,32,8))

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
        assert output[0].shape == (2, 32, 32, 16)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_2d_concat_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_position_embedding_2d_reproject():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding2D
    layer = PositionEmbedding2D(concat=True, reproject_embedding=True)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,32,32,8))

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
        assert output[0].shape == (2, 32, 32, 8)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_2d_concat_reproject_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_position_embedding_3d_vanilla():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding3D
    layer = PositionEmbedding3D()

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,8,8,8,24))

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
        assert output[0].shape == (2,8,8,8,24)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_3d_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_position_embedding_3d_concat():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding3D
    layer = PositionEmbedding3D(concat=True)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,6,6,6,24))

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
        assert output[0].shape == (2, 6, 6, 6, 48)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_3d_concat_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_position_embedding_3d_reproject():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import PositionEmbedding3D
    layer = PositionEmbedding3D(concat=True, reproject_embedding=True)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,6,6,6,24))

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
        assert output[0].shape == (2, 6, 6, 6, 24)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('position_embedding_3d_concat_reproject_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_learned_embedding_vanilla():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import LearnedEmbedding
    layer = LearnedEmbedding()

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,8,24))

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
        assert output[0].shape == (2, 8, 24)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('learned_embedding_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)

def test_learned_embedding_concat():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.layers.position_embedding import LearnedEmbedding
    layer = LearnedEmbedding(concat=True)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((2,8,24))

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
        assert output[0].shape == (2, 8, 48)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('learned_embedding_concat_expected_output',
                     output, __file__, 'regression_outputs/test_position_embedding_outputs.json',
                     debug=_RK_REBUILD_REGRESSION)