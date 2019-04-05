from rinokeras import RK_USE_TF_VERSION as _RK_USE_TF_VERSION

if _RK_USE_TF_VERSION == 1:
    from rinokeras.core.v1x.utils import load_distributed
    from rinokeras.core.v1x.utils import convert_padding_mask_to_attention_mask
    from rinokeras.core.v1x.utils import convert_sequence_length_to_sequence_mask
    from rinokeras.core.v1x.utils import convert_sequence_mask_to_attention_mask
    from rinokeras.core.v1x.utils import convert_to_attention_mask
    from rinokeras.core.v1x.utils import MetricsAccumulator
    from rinokeras.core.v1x.utils import accuracy
    from rinokeras.core.v1x.utils import bleu1
    from rinokeras.core.v1x.utils import bleu2
    from rinokeras.core.v1x.utils import bleu3
    from rinokeras.core.v1x.utils import bleu4
    from rinokeras.core.v1x.utils import rouge_l
    from rinokeras.core.v1x.utils import Gradients
    from rinokeras.core.v1x.utils import clip_gradients
    from rinokeras.core.v1x.utils import get_optimizer
    from rinokeras.core.v1x.utils import gather_from_last
    from rinokeras.core.v1x.utils import get_shape
elif _RK_USE_TF_VERSION == 2:
    pass
