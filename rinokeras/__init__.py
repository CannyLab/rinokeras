import tensorflow as tf
from packaging import version

RK_USE_TF_VERSION = None
if version.parse(tf.__version__) < version.parse("2.0.0a0"):
    RK_USE_TF_VERSION = 1
    import rinokeras.compat
    import rinokeras.layers
    import rinokeras.models
    import rinokeras.train
    import rinokeras.utils
else:
    RK_USE_TF_VERSION = 2
    import rinokeras.compat
    import rinokeras.layers
    import rinokeras.models

