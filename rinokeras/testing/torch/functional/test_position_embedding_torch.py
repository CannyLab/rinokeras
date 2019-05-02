
import torch
import numpy as np

def test_position_embedding_no_concat_1d():
    from rinokeras.torch.functional import position_embed

    pre_embed = torch.randn(2, 4, 8, dtype=torch.float32)
    post_embed = position_embed(pre_embed, start=1, concat=False)

    pre_numpy = pre_embed.numpy()
    post_numpy = post_embed.numpy()

    assert post_embed.shape == (2, 4, 8)
    assert not np.isclose(pre_numpy-post_numpy, np.zeros([2,4,8])).any()

def test_position_embedding_concat_1d():
    from rinokeras.torch.functional import position_embed

    pre_embed = torch.randn(2, 4, 8, dtype=torch.float32)
    post_embed = position_embed(pre_embed, start=1, concat=True)

    assert post_embed.shape == (2, 4, 16)

    



