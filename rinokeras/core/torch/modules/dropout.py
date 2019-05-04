
from typing import Optional
import torch

class LayerDropout(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, dropout_probability: Optional[float]=None) -> None:
        """
        Layer dropout - with probability dropout_probability keeps the layer, and 
        with probability 1-dropout_probability drops the layer alltogether. From
        https://arxiv.org/abs/1603.09382
        
        Arguments:
            layer {torch.nn.Module} -- The layer to wrap
        
        Keyword Arguments:
            dropout_probability {Optional[float]} -- The probability with which to drop 
                the layer (default: {None})
        
        Raises:
            ValueError: If the keep probability is outside [0,1]
        """

        super(LayerDropout, self).__init__()
        self.keep_probability = dropout_probability
        if self.dropout_probability and ( self.dropout_probability < 0 or self.dropout_probability > 1):
            raise ValueError('Keep probability in layer dropout should be in [0,1]')
        self.layer = layer

    def forward(self, *inputs):
        if self.dropout_probability and torch.rand([1]).item() < self.dropout_probability:
            return inputs
        return self.layer(*inputs)
