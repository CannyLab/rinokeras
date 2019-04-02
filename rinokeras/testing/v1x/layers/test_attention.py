
import tensorflow as tf
import numpy as np
import json
import os
import warnings

from rinokeras.testing import RK_REBUILD_REGRESSION_TESTS as _RK_REBUILD_REGRESSION
from rinokeras.testing.utils import check_regression

def check_from_config(__class, __obj):
    assert __class.from_config(__obj.get_config()) is not None


def test_luongAttention():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import LuongAttention
    luong_attention_layer = LuongAttention(local=False, stddev=1.0,
                                           regularizer=None)
    assert luong_attention_layer is not None

    # Encoded values
    encoded_values = np.random.sample((16, 10, 128))
    query_values = np.random.sample((16, 128))

    # Get some sample input tensors
    encoder_tensor = tf.constant(encoded_values)  # BS x SEQLEN x ENC_CELL_SIZE
    query_tensor = tf.constant(query_values)  # BS x DEC_CELL_SIZE

    value = luong_attention_layer((query_tensor, encoder_tensor))

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output is not None  # Make sure the value is not none
    assert output.shape == (16, 128)  # Make sure the output shape is correct

    # Do regression testing
    check_regression('luong_attention_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_luongAttention_local():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import LuongAttention
    luong_attention_layer = LuongAttention(local=True, stddev=1.0,
                                           regularizer=None)
    assert luong_attention_layer is not None

    # Encoded values
    encoded_values = np.random.sample((16, 10, 128))
    query_values = np.random.sample((16, 128))
    position_values = np.random.randint(0, 10, (16,))

    # Get some sample input tensors
    encoder_tensor = tf.constant(encoded_values)  # BS x SEQLEN x ENC_CELL_SIZE
    query_tensor = tf.constant(query_values)  # BS x DEC_CELL_SIZE
    position_tensor = tf.constant(position_values)  # BS

    value = luong_attention_layer((query_tensor, encoder_tensor,
                                   position_tensor))

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output is not None  # Make sure the value is correct
    assert output.shape == (16, 128)  # Make sure the output shape is correct

    # Do regression testing
    check_regression('luong_attention_local_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_attentionQKVProjection():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import AttentionQKVProjection
    attention_qkv_projection = AttentionQKVProjection(key_depth=8,
                                                      value_depth=12)
    assert attention_qkv_projection is not None

    # Encoded values
    query_values = np.random.sample((16, 10, 128))
    key_values = np.random.sample((16, 5, 128))
    value_values = np.random.sample((16, 5, 128))

    # Get some sample input tensors
    query_tensor = tf.constant(query_values)
    key_tensor = tf.constant(key_values)
    value_tensor = tf.constant(value_values)

    value = attention_qkv_projection((query_tensor, key_tensor, value_tensor))

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output is not None  # Make sure the value is correct
    # Make sure the output shape is correct
    assert output[0].shape == (16, 10, 8)
    # Make sure the output shape is correct
    assert output[1].shape == (16, 5, 8)
    # Make sure the output shape is correct
    assert output[2].shape == (16, 5, 12)

    # Do regression testing
    check_regression('attentionqkv_projection_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_trilinearSimilarity():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import TrilinearSimilarity
    trilinear_similarity_layer = TrilinearSimilarity()
    assert trilinear_similarity_layer is not None

    # Encoded values
    query_values = np.random.sample((16, 10, 128))
    context_values = np.random.sample((16, 5, 128))

    # Get some sample input tensors
    query_tensor = tf.constant(query_values)
    context_tensor = tf.constant(context_values)

    value = trilinear_similarity_layer((context_tensor, query_tensor))

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output is not None  # Make sure the value is correct
    assert output.shape == (16, 5, 10)  # Make sure the output shape is correct

    # Do regression testing
    check_regression('trilinear_similarity_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_scaledDotProductSimilarity():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import ScaledDotProductSimilarity
    sdp_layer = ScaledDotProductSimilarity()
    assert sdp_layer is not None

    # Encoded values
    query_values = np.random.sample((16, 10, 128))
    context_values = np.random.sample((16, 5, 128))

    # Get some sample input tensors
    query_tensor = tf.constant(query_values)
    context_tensor = tf.constant(context_values)

    value = sdp_layer((context_tensor, query_tensor))

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output is not None  # Make sure the value is correct
    assert output.shape == (16, 5, 10)  # Make sure the output shape is correct

    # Do regression testing
    check_regression('scaled_dot_product_similarity_expected_output', 
                     output, __file__, 'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_applyAttentionMask():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import ApplyAttentionMask
    aam_layer = ApplyAttentionMask()
    assert aam_layer is not None

    # Encoded values
    similarity_values = np.ones((16, 10, 10))
    similarity_values_heads = np.ones((16, 4, 10, 10))
    mask_values = np.random.choice([0, 1], size=(16, 10, 10))
    mask_values_heads = np.random.choice([0, 1], size=(16, 10, 10))

    # Get some sample input tensors
    similarity_tensor = tf.constant(similarity_values)
    similarity_heads_tensor = tf.constant(similarity_values_heads)
    mask_tensor = tf.constant(mask_values)
    mask_heads_tensor = tf.constant(mask_values_heads)

    value = aam_layer(inputs=similarity_tensor, mask=mask_tensor)
    value_heads = aam_layer(inputs=similarity_heads_tensor,
                            mask=mask_heads_tensor)

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run([value, value_heads])

    assert output[0] is not None  # Make sure the value is not none
    assert output[1] is not None  # Make sure the value is not none
    assert output[0].shape == (16, 10, 10)  # Make sure the value is not none
    # Make sure the value is not none
    assert output[1].shape == (16, 4, 10, 10)

    check_regression('apply_attention_mask_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_attentionMap():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import AttentionMap
    from rinokeras.core.v1x.common.attention import ScaledDotProductSimilarity
    sdp = ScaledDotProductSimilarity()
    attention_map = AttentionMap(similarity_metric=sdp,
                                 attention_function=tf.nn.softmax)
    assert attention_map is not None
    assert sdp is not None

    # Encoded values
    query_values = np.random.sample((16, 8, 12))
    key_values = np.random.sample((16, 20, 12))
    value_values = np.random.sample((16, 20, 12))
    mask_values = np.random.choice([0, 1], size=(16, 8, 20))

    # Get some sample input tensors
    query_tensor = tf.constant(query_values)
    key_tensor = tf.constant(key_values)
    value_tensor = tf.constant(value_values)
    mask_tensor = tf.constant(mask_values)

    value = attention_map(inputs=(query_tensor, key_tensor, value_tensor),
                          mask=mask_tensor)

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output[0] is not None  # Make sure the value is not none
    assert output[1] is not None  # Make sure the value is not none
    assert output[0].shape == (16, 8, 12)
    assert output[1].shape == (16, 8, 20)

    check_regression('attention_map_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_multiHeadAttentionMap():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import MultiHeadAttentionMap
    from rinokeras.core.v1x.common.attention import ScaledDotProductSimilarity
    sdp = ScaledDotProductSimilarity()
    attention_map = MultiHeadAttentionMap(similarity_metric=sdp,
                                          n_heads=4,
                                          attention_function=tf.nn.softmax)
    assert attention_map is not None
    assert sdp is not None

    # Encoded values
    query_values = np.random.sample((16, 8, 12))
    key_values = np.random.sample((16, 20, 12))
    value_values = np.random.sample((16, 20, 12))
    mask_values = np.random.choice([0, 1], size=(16, 8, 20))

    # Get some sample input tensors
    query_tensor = tf.constant(query_values)
    key_tensor = tf.constant(key_values)
    value_tensor = tf.constant(value_values)
    mask_tensor = tf.constant(mask_values)

    value = attention_map(inputs=(query_tensor, key_tensor, value_tensor),
                          mask=mask_tensor,
                          return_attention_weights=True)

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output[0] is not None  # Make sure the value is not none
    assert output[1] is not None  # Make sure the value is not none
    assert output[0].shape == (16, 8, 12)
    assert output[1].shape == (16, 4, 8, 20)

    masked_vals = np.squeeze(output[1][:, 0, :, :])[np.where(mask_values == 0)]
    assert np.isclose(masked_vals, np.zeros_like(masked_vals)).all()

    check_regression('multihead_attention_map_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_multiHeadAttention():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import MultiHeadAttention
    attention_map = MultiHeadAttention(similarity_metric='scaled_dot',
                                       n_heads=4)
    assert attention_map is not None

    # Encoded values
    query_values = np.random.sample((16, 8, 12))
    key_values = np.random.sample((16, 20, 12))
    value_values = np.random.sample((16, 20, 12))
    mask_values = np.random.choice([0, 1], size=(16, 8, 20))

    # Get some sample input tensors
    query_tensor = tf.constant(query_values)
    key_tensor = tf.constant(key_values)
    value_tensor = tf.constant(value_values)
    mask_tensor = tf.constant(mask_values)

    value = attention_map(inputs=(query_tensor, key_tensor, value_tensor),
                          mask=mask_tensor,
                          return_attention_weights=True)

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output[0] is not None  # Make sure the value is not none
    assert output[1] is not None  # Make sure the value is not none
    assert output[0].shape == (16, 8, 12)
    assert output[1].shape == (16, 4, 8, 20)

    # Check the masking
    masked_vals = np.squeeze(output[1][:, 0, :, :])[np.where(mask_values == 0)]
    assert np.isclose(masked_vals, np.zeros_like(masked_vals)).all()

    check_regression('multihead_attention_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)

    check_from_config(MultiHeadAttention, attention_map)


def test_multiHeadAttention_trilinear():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import MultiHeadAttention
    attention_map = MultiHeadAttention(similarity_metric='trilinear',
                                       n_heads=4)
    assert attention_map is not None

    # Encoded values
    query_values = np.random.sample((16, 8, 12))
    key_values = np.random.sample((16, 20, 12))
    value_values = np.random.sample((16, 20, 12))
    mask_values = np.random.choice([0, 1], size=(16, 8, 20))

    # Get some sample input tensors
    query_tensor = tf.constant(query_values)
    key_tensor = tf.constant(key_values)
    value_tensor = tf.constant(value_values)
    mask_tensor = tf.constant(mask_values)

    value = attention_map(inputs=(query_tensor, key_tensor, value_tensor),
                          mask=mask_tensor,
                          return_attention_weights=True)

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output[0] is not None  # Make sure the value is not none
    assert output[1] is not None  # Make sure the value is not none
    assert output[0].shape == (16, 8, 12)
    assert output[1].shape == (16, 4, 8, 20)

    # Check the masking
    masked_vals = np.squeeze(output[1][:, 0, :, :])[np.where(mask_values == 0)]
    assert np.isclose(masked_vals, np.zeros_like(masked_vals)).all()

    check_regression('multihead_attention_trilinear_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)


def test_selfAttention():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import SelfAttention
    attention_map = SelfAttention(similarity_metric='scaled_dot', n_heads=4)
    assert attention_map is not None

    # Encoded values
    sa_values = np.random.sample((4, 128, 12))
    mask_values = np.random.choice([0, 1], size=(4, 128, 128))

    # Get some sample input tensors
    sa_tensor = tf.constant(sa_values)
    mask_tensor = tf.constant(mask_values)

    value = attention_map(inputs=sa_tensor,
                          mask=mask_tensor,
                          return_attention_weights=True)

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output[0] is not None  # Make sure the value is not none
    assert output[1] is not None  # Make sure the value is not none
    assert output[0].shape == (4, 128, 12)
    assert output[1].shape == (4, 4, 128, 128)

    # WARNING: This might fail because probability
    masked_vals = np.squeeze(output[1][:, 0, :, :])[np.where(mask_values == 0)]
    assert np.isclose(masked_vals, np.zeros_like(masked_vals)).all()

    check_regression('self_attention_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)

    # Check that you can instantiate a layer from the config
    check_from_config(SelfAttention, attention_map)


def test_contextQueryAttention():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.core.v1x.common.attention import ContextQueryAttention
    attention_map = ContextQueryAttention(similarity_metric='trilinear')
    assert attention_map is not None

    # Encoded values
    context_values = np.random.sample((16, 8, 12))
    query_values = np.random.sample((16, 10, 12))
    mask_values = np.random.choice([0, 1], size=(16, 8, 10))

    # Get some sample input tensors
    context_tensor = tf.constant(context_values)
    query_tensor = tf.constant(query_values)
    mask_tensor = tf.constant(mask_values)

    value = attention_map(inputs=(context_tensor, query_tensor),
                          mask=mask_tensor)

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    assert output is not None  # Make sure the value is not none
    assert output.shape == (16, 8, 4*12)

    check_regression('context_query_attention_expected_output', output, __file__,
                     'regression_outputs/test_attention_outputs.json', debug=_RK_REBUILD_REGRESSION)
