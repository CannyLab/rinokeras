"""
Testing for the transformer
"""
import tensorflow as tf
import tempfile

from rinokeras.core.v1x.utils import convert_to_attention_mask
from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import reset_session, random_tensor, run_simple_session,\
        assert_not_none, assert_expected_shapes, check_regression, \
        random_mask_tensor, random_sequence_tensor


def test_convert_sequence_mask_to_attention_mask():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.utils import convert_sequence_mask_to_attention_mask
    # Encoded values
    input_tensor, _ = random_sequence_tensor(2, 5, 6)
    input_mask = tf.convert_to_tensor([[1,1,1,0,0],[1,1,1,1,1]])

    # Get the output of the layer
    value = convert_sequence_mask_to_attention_mask(input_tensor, input_mask)
    # Construct the session
    output = run_simple_session(inputs=[value], feed={})
    assert_not_none(output)
    assert_expected_shapes(output,[(2,5,5)])
    # Do regression testing
    check_regression('convert_sequence_mask_to_attention_mask',
                     output, __file__, 'regression_outputs/test_utils_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_convert_sequence_length_to_sequence_mask():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.utils import convert_sequence_length_to_sequence_mask
    # Encoded values
    input_tensor, _ = random_sequence_tensor(2, 32, 128)
    input_mask = tf.convert_to_tensor([17,9])

    # Get the output of the layer
    value = convert_sequence_length_to_sequence_mask(input_tensor, input_mask)
    # Construct the session
    output = run_simple_session(inputs=[value], feed={})
    assert_not_none(output)
    assert_expected_shapes(output,[(2,32)])
    # Do regression testing
    check_regression('convert_sequence_length_to_sequence_mask',
                     output, __file__, 'regression_outputs/test_utils_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_convert_to_attention_mask_1():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.utils import convert_to_attention_mask
    # Encoded values
    input_tensor, _ = random_sequence_tensor(2, 32, 128)
    input_mask = tf.convert_to_tensor([17,9])

    # Get the output of the layer
    value = convert_to_attention_mask(input_tensor, input_mask)
    # Construct the session
    output = run_simple_session(inputs=[value], feed={})
    assert_not_none(output)
    assert_expected_shapes(output,[(2,32,32)])
    # Do regression testing
    check_regression('convert_to_attention_mask_1',
                     output, __file__, 'regression_outputs/test_utils_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_convert_to_attention_mask_2():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.utils import convert_to_attention_mask
    # Encoded values
    input_tensor, _ = random_sequence_tensor(2, 32, 128)
    input_mask, _ = random_mask_tensor(2,32)

    # Get the output of the layer
    value = convert_to_attention_mask(input_tensor, input_mask)
    # Construct the session
    output = run_simple_session(inputs=[value], feed={})
    assert_not_none(output)
    assert_expected_shapes(output,[(2,32,32)])
    # Do regression testing
    check_regression('convert_to_attention_mask_2',
                     output, __file__, 'regression_outputs/test_utils_outputs.json', debug=_RK_REBUILD_REGRESSION)
