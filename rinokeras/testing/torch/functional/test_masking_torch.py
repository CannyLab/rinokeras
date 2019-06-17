
import torch
import numpy as np

from rinokeras.testing.torch.utils import random_mask_tensor

def test_apply_attention_mask_softmax():

    from rinokeras.torch.functional import apply_attention_mask

    inputs = torch.randn(2,4,32)
    mask = random_mask_tensor(2, 4, 32)
    masked = apply_attention_mask(inputs, mask, hadamard=False)
    masked = torch.nn.functional.softmax(masked, dim=-1)

    tv1 = np.where(mask.numpy(), np.zeros_like(masked.numpy()), masked.numpy())
    tv2 = np.zeros_like(tv1)
    assert np.isclose(tv1, tv2).all()

def test_apply_attention_mask_hadamard():

    from rinokeras.torch.functional import apply_attention_mask

    inputs = torch.randn(2,4,32)
    mask = random_mask_tensor(2, 4, 32)
    masked = apply_attention_mask(inputs, mask, hadamard=True)

    tv1 = np.where(mask.numpy(), np.zeros_like(masked.numpy()), masked.numpy())
    tv2 = np.zeros_like(tv1)
    assert np.isclose(tv1, tv2).all()
