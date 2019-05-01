"""
Utilities for torch tensors
"""

import torch

def get_variable(*shapes,
                 initializer='xavier_normal',
                 dtype=None,
                 layout=torch.strided,
                 device=None,
                 requires_grad=True,
                 pin_memory=False) -> torch.Tensor:
    initializer_map = {
        'xavier_uniform': torch.nn.init.xavier_uniform,
        'xavier_normal': torch.nn.init.xavier_normal,
    }
    if initializer not in initializer_map:
        raise KeyError('Initializer type {} unrecognized.'.format(initializer))

    return initializer_map[initializer](torch.empty(shapes,
                                                    dtype=dtype,
                                                    layout=layout,
                                                    device=device,
                                                    requires_grad=requires_grad,
                                                    pin_memory=pin_memory))

def get_parameter(*args, **kwargs):
    return torch.nn.Parameter(get_variable(*args, **kwargs))
