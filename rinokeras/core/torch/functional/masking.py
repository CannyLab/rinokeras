
import torch

from rinokeras.core.torch.utils.numbers import inf

def apply_attention_mask(inputs: torch.Tensor, mask: torch.Tensor = None, hadamard: bool = False) -> torch.Tensor:
    if mask is None:
        return inputs
    
    # Check tensor ranks
    if len(inputs.shape) not in (3,4) or len(mask.shape) != 3:
        raise AssertionError('Rank not correct {}, {} for apply_attention_mask'.format(
            inputs.shape, mask.shape))
    
    # We need to expand the dimensions
    if len(mask.shape) != len(inputs.shape):
        mask = mask.unsqueeze(-1)
        if mask.shape != inputs.shape:
            raise AssertionError('Mask shape {} not compatible with input shape {}'.format(
                mask.shape, inputs.shape))
    
    if hadamard:
        # Just directly multiply the mask and the inputs
        return mask.byte().float() * inputs

    # Otherwise return the mask * a large number added to the inputs
    return -inf * (1-mask.byte().float()) + inputs
