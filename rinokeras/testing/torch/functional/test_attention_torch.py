"""
Test some of the attention code
"""

import torch
import time

def test_attention_map_base():
    from rinokeras.torch.functional import attention_map

    # Generate some random tensor data
    queries_a = torch.randn(1, 4, 3)
    keys_a = queries_a
    values_a = queries_a

    output = attention_map(queries_a, keys_a, values_a)

    assert output[0].shape == (1, 4, 3)

def test_multi_head_attention_map_base():
    from rinokeras.torch.functional import multi_head_attention_map

    # Generate some random tensor data
    queries_a = torch.randn(8, 16, 4)
    keys_a = torch.randn(8, 3, 4)
    values_a = torch.randn(8, 3, 12)

    output = multi_head_attention_map(queries_a, keys_a, values_a, 4)

    assert output[0].shape == (8, 16, 12)
    assert output[1].shape == (8, 4, 16, 3)
    