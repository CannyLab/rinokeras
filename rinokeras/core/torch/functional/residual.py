"""
Functional residual layers
"""

import torch

def residual(layer, *inputs):
    """
    Computes a simple residual F(x) + x
    """
    return inputs + layer(*inputs)

def highway(layer, inputs, gate_weights):
    """
    Computes a weighted residual Gx*F(x) + (1-Gx)x
    """
    layer_out = layer(inputs)
    gated_out = torch.nn.functional.linear(inputs, gate_weights)
    return gated_out * layer_out + (torch.ones_like(gated_out) - gated_out) * inputs