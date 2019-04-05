from rinokeras import RK_USE_TF_VERSION as _RK_USE_TF_VERSION

if _RK_USE_TF_VERSION == 1:
    from rinokeras.core.v1x.train import Experiment
    from rinokeras.core.v1x.train import TestGraph
    from rinokeras.core.v1x.train import TrainGraph
elif _RK_USE_TF_VERSION == 2:
    pass
