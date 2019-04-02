from rinokeras import RK_USE_TF_VERSION as _RK_USE_TF_VERSION

if _RK_USE_TF_VERSION == 1:
    from rinokeras.core.v1x.models.qanet import *
elif _RK_USE_TF_VERSION == 2:
    raise NotImplementedError('QANet not yet supported in RK2.0')
