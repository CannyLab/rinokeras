from rinokeras import RK_USE_TF_VERSION as _RK_USE_TF_VERSION

if _RK_USE_TF_VERSION == 1:
    from rinokeras.core.v1x.models.transformer.transformer_attention import \
        TransformerMultiAttention, TransformerSelfAttention
    from rinokeras.core.v1x.models.transformer.transformer_ff import \
        TransformerFeedForward
    from rinokeras.core.v1x.models.transformer.transformer_embedding import \
        TransformerInputEmbedding
    from rinokeras.core.v1x.models.transformer.transformer_encoder import \
        TransformerEncoderBlock, TransformerEncoder
    from rinokeras.core.v1x.models.transformer.transformer_decoder import \
        TransformerDecoderBlock, TransformerDecoder
    from rinokeras.core.v1x.models.transformer.transformer import \
        Transformer
elif _RK_USE_TF_VERSION == 2:
    raise NotImplementedError('Transformer not yet supported in RK2.0')
