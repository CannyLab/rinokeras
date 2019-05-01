
from typing import Optional
import torch

class LayerDropout(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, keep_probability: Optional[float]=None):
        """
        Layer dropout - with probability keep_probability keeps the layer, and 
        with probability 1-keep_probability drops the layer alltogether. From
        https://arxiv.org/abs/1603.09382
        
        Arguments:
            layer {torch.nn.Module} -- The layer to wrap
        
        Keyword Arguments:
            keep_probability {Optional[float]} -- The probability with which to keep 
                the layer (default: {None})
        
        Raises:
            ValueError: If the keep probability is outside [0,1]
        """

        super(LayerDropout, self).__init__()
        self.keep_probability = keep_probability
        if self.keep_probability and ( self.keep_probability < 0 or self.keep_probability > 1):
            raise ValueError('Keep probability in layer dropout should be in [0,1]')
        self.layer = layer

    def forward(self, *inputs):
        if self.keep_probability and torch.rand([1]).item() > self.keep_probability:
            return inputs
        return self.layer(*inputs)