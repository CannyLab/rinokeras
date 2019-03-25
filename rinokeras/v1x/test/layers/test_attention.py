
import tensorflow as tf
import numpy as np
import json
import os

def get_local_file(fpath):
    return '/'+os.path.join(*__file__.split(os.sep)[:-1], fpath)

def test_luongAttention():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.v1x.common.attention import LuongAttention
    luong_attention_layer = LuongAttention(local=False, stddev=1.0, regularizer=None)
    assert luong_attention_layer is not None

    # Encoded values
    encoded_values = np.random.sample((16, 10, 128))
    query_values = np.random.sample((16, 128))

    # Get some sample input tensors
    encoder_tensor = tf.constant(encoded_values) # BS x SEQLEN x ENC_CELL_SIZE
    query_tensor = tf.constant(query_values) # BS x DEC_CELL_SIZE

    value = luong_attention_layer((query_tensor, encoder_tensor))

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    with open(get_local_file('test_attention_outputs.json'), 'r') as json_file:
        jf = json.loads(json_file.read())
        expected_output = np.array(jf['luong_attention_expected_output'])    
    
    assert output is not None  # Make sure the value is correct
    assert output.shape == (16, 128)  # Make sure the output shape is correct
    assert np.isclose(output, expected_output).all()

def test_luongAttention_local():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.v1x.common.attention import LuongAttention
    luong_attention_layer = LuongAttention(local=True, stddev=1.0, regularizer=None)
    assert luong_attention_layer is not None

    # Encoded values
    encoded_values = np.random.sample((16, 10, 128))
    query_values = np.random.sample((16, 128))
    position_values = np.random.randint(0, 10, (16,))

    # Get some sample input tensors
    encoder_tensor = tf.constant(encoded_values) # BS x SEQLEN x ENC_CELL_SIZE
    query_tensor = tf.constant(query_values) # BS x DEC_CELL_SIZE
    position_tensor = tf.constant(position_values) # BS

    value = luong_attention_layer((query_tensor, encoder_tensor, position_tensor))

    # Construct the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    with open(get_local_file('test_attention_outputs.json'), 'r') as json_file:
        jf = json.loads(json_file.read())
        expected_output = np.array(jf['luong_attention_local_expected_output'])    
    
    assert output is not None  # Make sure the value is correct
    assert output.shape == (16, 128)  # Make sure the output shape is correct
    assert np.isclose(output, expected_output).all()

def test_attentionQKVProjection():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.v1x.common.attention import AttentionQKVProjection
    attention_qkv_projection = AttentionQKVProjection(key_depth=8, value_depth=12)
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
    config.gpu_options.allow_growth=True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    with open(get_local_file('test_attention_outputs.json'), 'r') as json_file:
        jf = json.loads(json_file.read())
    jf['attention_qkv_projection_expected_output'] = [i.tolist() for i in output]
    with open(get_local_file('test_attention_outputs.json'), 'w') as json_file:
        json.dump(jf, json_file)
    with open(get_local_file('test_attention_outputs.json'), 'r') as json_file:
        jf = json.loads(json_file.read())
        expected_output = [np.array(v) for v in jf['attention_qkv_projection_expected_output']] 
    
    assert output is not None  # Make sure the value is correct
    assert output[0].shape == (16, 10, 8)  # Make sure the output shape is correct
    assert output[1].shape == (16, 5, 8)  # Make sure the output shape is correct
    assert output[2].shape == (16, 5, 12)  # Make sure the output shape is correct
    assert np.isclose(output[0], expected_output[0]).all()
    assert np.isclose(output[1], expected_output[1]).all()
    assert np.isclose(output[2], expected_output[2]).all()

def test_trilinearSimilarity():
    tf.reset_default_graph()
    np.random.seed(256)
    tf.random.set_random_seed(256)
    # Construct the layer
    from rinokeras.v1x.common.attention import TrilinearSimilarity
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
    config.gpu_options.allow_growth=True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(value)

    # with open(get_local_file('test_attention_outputs.json'), 'r') as json_file:
    #     jf = json.loads(json_file.read())
    # jf['trilinear_similarity_expected_output'] = output.tolist()
    # with open(get_local_file('test_attention_outputs.json'), 'w') as json_file:
    #     json.dump(jf, json_file)
    with open(get_local_file('test_attention_outputs.json'), 'r') as json_file:
        jf = json.loads(json_file.read())
        expected_output = np.array(jf['trilinear_similarity_expected_output'])
    
    assert output is not None  # Make sure the value is correct
    assert output.shape == (16, 5, 10)  # Make sure the output shape is correct
    assert np.isclose(output, expected_output).all() 

def test_scaledDotProductSimilarity():
    pass


def test_applyAttentionMask():
    pass


def test_attentionMap():
    pass


def test_multiHeadAttentionMap():
    pass


def test_multiHeadAttention():
    pass


def test_selfAttention():
    pass


def test_contextQueryAttention():
    pass

