
from typing import Type
import torch


class Residual(torch.nn.Module):
    """
    Simple residual wrapper
    """

    def __init__(self, layer: Type[torch.nn.Module]):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, inputs, *args, **kwargs):
        return inputs + self.layer(inputs, *args, **kwargs)

