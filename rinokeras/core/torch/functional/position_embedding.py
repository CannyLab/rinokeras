"""
Position Embedding code, based on:
https://arxiv.org/pdf/1706.03762.pdf
"""

import torch

def position_embed(inputs: torch.Tensor, start: int = 1, concat: bool = False, base: int = 10000) -> torch.Tensor:
    hidden_size = inputs.shape[-1]
    if concat and hidden_size % 2 != 0:
        raise AssertionError('Model hidden size must be even for sinusoidal embedding')
    elif hidden_size % 2 != 0:
        hidden_size += 1

    if inputs.is_cuda:
        power = torch.arange(0, hidden_size, 2).float().cuda() / hidden_size
        seqpos = torch.arange(start, inputs.shape[1]+1).unsqueeze(0).float().cuda()
    else:
        power = torch.arange(0, hidden_size, 2).float() / hidden_size
        seqpos = torch.arange(start, inputs.shape[1]+1).unsqueeze(0).float()

    divisor = base ** power

    # Compute the sequence positions
    index = seqpos.unsqueeze(-1) / divisor
    
    sin_embedding = torch.sin(index)
    cos_embedding = torch.cos(index)
    
    position_embedding = torch.stack([sin_embedding, cos_embedding], dim=-1).reshape(1, inputs.shape[1], hidden_size)

    if concat:
        output = torch.cat([inputs, position_embedding.expand(inputs.shape[0], -1, -1)], dim=-1)
        return output
    return inputs + position_embedding
