# We only support a single version of torch, so these are all the 
# same

from rinokeras.core.torch.modules.residual import Residual
from rinokeras.core.torch.modules.activations import GatedTanh
from rinokeras.core.torch.modules.dropout import LayerDropout
from rinokeras.core.torch.modules.attention import MultiHeadAttention, SelfAttention, StridedCachedLWSelfAttention