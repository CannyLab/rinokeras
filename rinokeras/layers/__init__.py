from rinokeras import RK_USE_TF_VERSION as _RK_USE_TF_VERSION

if _RK_USE_TF_VERSION == 1:
    # Activations
    from rinokeras.core.v1x.common import GatedTanh

    # Autoregressive
    from rinokeras.core.v1x.common import RandomGaussNoise
    from rinokeras.core.v1x.common import CouplingLayer

    # Conv
    from rinokeras.core.v1x.common import NormedConvStack
    from rinokeras.core.v1x.common import ResidualBlock
    from rinokeras.core.v1x.common import GroupedConvolution

    # Dropout
    from rinokeras.core.v1x.common import LayerDropout

    # Inversion
    from rinokeras.core.v1x.common import DenseTranspose
    from rinokeras.core.v1x.common import EmbeddingTranspose
    from rinokeras.core.v1x.common import InvertibleDense

    # Masking
    from rinokeras.core.v1x.common import BERTRandomReplaceMask

    # Normalization
    from rinokeras.core.v1x.common import LayerNorm
    from rinokeras.core.v1x.common import WeightNormDense

    # Embedding
    from rinokeras.core.v1x.common import PositionEmbedding
    from rinokeras.core.v1x.common import PositionEmbedding2D
    from rinokeras.core.v1x.common import PositionEmbedding3D
    from rinokeras.core.v1x.common import LearnedEmbedding

    # Residual
    from rinokeras.core.v1x.common import Residual
    from rinokeras.core.v1x.common import Highway

    # Stack
    from rinokeras.core.v1x.common import Stack
    from rinokeras.core.v1x.common import Conv2DStack
    from rinokeras.core.v1x.common import Deconv2DStack
    from rinokeras.core.v1x.common import DenseStack

    # Attention
    from rinokeras.core.v1x.common import LuongAttention
    from rinokeras.core.v1x.common import AttentionQKVProjection
    from rinokeras.core.v1x.common import TrilinearSimilarity
    from rinokeras.core.v1x.common import ScaledDotProductSimilarity
    from rinokeras.core.v1x.common import ApplyAttentionMask
    from rinokeras.core.v1x.common import AttentionMap
    from rinokeras.core.v1x.common import MultiHeadAttentionMap
    from rinokeras.core.v1x.common import MultiHeadAttention
    from rinokeras.core.v1x.common import SelfAttention
    from rinokeras.core.v1x.common import ContextQueryAttention

    # GCN
    from rinokeras.core.v1x.common import GraphConvolutionalLayer
elif _RK_USE_TF_VERSION == 2:
    pass
