
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()  
np.random.seed(182)
tf.set_random_seed(182)


def test_luongAttention():
    
    from rinokeras.common.attention import LuongAttention
    
    # Get some testing data
    target_hidden = np.random.rand(5, 2).astype(np.float32)  # Target Hidden shape
    source_hidden_sequence = np.random.rand(5, 5, 2).astype(np.float32)

    # Test non-local, stddev 1
    test_attention_map = LuongAttention(local=False, stddev=1.0)

    # Apply the attention map
    output = test_attention_map((target_hidden, source_hidden_sequence))

    # Get the expected output
    expected_output = np.array([[-0.5965891, 0.33805916],
                                [-0.49869075, 0.48035246],
                                [-0.6700753, 0.38853297],
                                [-0.6930723, 0.4312901],
                                [-0.7170821, 0.48857006]], dtype=np.float32)
 
    if not np.allclose(output, expected_output):
        raise AssertionError("Luong Attention [Local=False, stddev=1.0] not as expected.")

    # Test non-local, stddev 0.5
    test_attention_map = LuongAttention(local=False, stddev=0.5)
    output = test_attention_map((target_hidden, source_hidden_sequence))

    expected_output = np.array([[-0.4305752, 0.10952884],
                                [0.30076617, -0.5722779],
                                [-0.36495835, -0.04738057],
                                [-0.2308154, -0.23675343],
                                [0.04679149, -0.53293025]], dtype=np.float32)

    if not np.allclose(output, expected_output):
        raise AssertionError("Luong Attention [Local=False, stddev=0.5] not as expected.")

    # Test local, stddev 1.0
    test_attention_map = LuongAttention(local=True, stddev=1.0)
    output = test_attention_map((target_hidden, source_hidden_sequence), t=3)

    expected_output = np.array([[0.27935886, 0.24433741],
                                [-0.6575557, 0.23246999],
                                [0.10390171, 0.23293544],
                                [-0.15431388, 0.31091437],
                                [-0.5740663, 0.39107955]], dtype=np.float32)

    if not np.allclose(output, expected_output):
        raise AssertionError("Luong Attention [Local=True, stddev=1.0] not as expected.")


def test_attentionQKV():
    pass


def test_trilinearSimilarity():
    from rinokeras.common.attention import TrilinearSimilarity
    
    # Just test the construction and passing random data through the layers
    layer = TrilinearSimilarity()
    layer = TrilinearSimilarity(regularizer=tf.keras.regularizers.l2)
    layer = TrilinearSimilarity(dropout=0.1, regularizer=tf.keras.regularizers.l2)


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

