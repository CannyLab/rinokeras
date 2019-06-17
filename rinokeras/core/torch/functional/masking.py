
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
        mask = mask.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1)
        if mask.shape != inputs.shape:
            raise AssertionError('Mask shape {} not compatible with input shape {}'.format(
                mask.shape, inputs.shape))
    
    if hadamard:
        # Just directly multiply the mask and the inputs
        return mask.byte().float() * inputs

    # Otherwise return the mask * a large number added to the inputs
    return -inf * (1-mask.byte().float()) + inputs

def convert_sequence_mask_to_attention_mask(sequence: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
    if sequence.shape[0] != sequence_mask.shape[0]:
        raise AssertionError('Batch size mismatch between sequence and sequence_mask')
    if len(sequence_mask.shape) != 2:
        raise AssertionError('Can only convert 2D sequence maskes to 3D attention masks')
    return sequence_mask.unsqueeze(1).expand(-1, sequence.shape[1], -1)

def convert_sequence_length_to_sequence_mask(sequence: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    if sequence.shape[0] != sequence_lengths.shape[0]:
        raise AssertionError('Batch size mismatch between sequence and sequence_lengths')
    if len(sequence_lengths.shape) != 1:
        raise AssertionError('Can only convert 1D sequence lengths to 2D sequence masks')
    
    if sequence.is_cuda:
        seqs = torch.arange(0, sequence.shape[1]).expand(sequence.shape[0], -1).cuda()
    else:
        seqs = torch.arange(0, sequence.shape[1]).expand(sequence.shape[0], -1)
    return torch.lt(seqs.float(), sequence_lengths.view(-1, 1).expand(-1, seqs.shape[1]).float()).float()
