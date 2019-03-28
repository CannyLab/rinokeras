from rinokeras import RK_USE_TF_VERSION as _RK_USE_TF_VERSION

if _RK_USE_TF_VERSION == 1:
    # Activations
    from rinokeras.v1x.common import GatedTanh

    # Autoregressive
    from rinokeras.v1x.common import RandomGaussNoise
    from rinokeras.v1x.common import CouplingLayer

    # Conv
    from rinokeras.v1x.common import NormedConv
    from rinokeras.v1x.common import ResidualBlock

    # Dropout
    from rinokeras.v1x.common import LayerDropout

    # Inversion
    from rinokeras.v1x.common import DenseTranspose
    from rinokeras.v1x.common import EmbeddingTranspose
    from rinokeras.v1x.common import InvertibleDense

    # Masking
    from rinokeras.v1x.common import MaskInput

    # Normalization
    from rinokeras.v1x.common import LayerNorm
    from rinokeras.v1x.common import WeightNormDense

    # Embedding
    from rinokeras.v1x.common import PositionEmbedding
    from rinokeras.v1x.common import PositionEmbedding2D
    from rinokeras.v1x.common import PositionEmbedding3D
    from rinokeras.v1x.common import LearnedEmbedding

    # Residual
    from rinokeras.v1x.common import Residual
    from rinokeras.v1x.common import Highway

    # Stack
    from rinokeras.v1x.common import Stack
    from rinokeras.v1x.common import Conv2DStack
    from rinokeras.v1x.common import Deconv2DStack
    from rinokeras.v1x.common import DenseStack

    # Attention
    from rinokeras.v1x.common import LuongAttention
    from rinokeras.v1x.common import AttentionQKVProjection
    from rinokeras.v1x.common import TrilinearSimilarity
    from rinokeras.v1x.common import ScaledDotProductSimilarity
    from rinokeras.v1x.common import ApplyAttentionMask
    from rinokeras.v1x.common import AttentionMap
    from rinokeras.v1x.common import MultiHeadAttentionMap
    from rinokeras.v1x.common import MultiHeadAttention
    from rinokeras.v1x.common import SelfAttention
    from rinokeras.v1x.common import ContextQueryAttention

    # GCN
    from rinokeras.v1x.common import GraphConvolutionalLayer
elif _RK_USE_TF_VERSION == 2:
    raise NotImplementedError('Layers not yet supported in RK2.0')
