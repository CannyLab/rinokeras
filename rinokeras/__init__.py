TF_NOT_FOUND = False
TORCH_NOT_FOUND = False

# Handle Tensorflow imports
try:
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
except ImportError as e:
    if 'tensorflow' not in str(e):
        raise

    TF_NOT_FOUND = True

# Handle Torch imports
try:
    import torch

    import rinokeras.torch.modules
    import rinokeras.torch.functional
    import rinokeras.torch.models

except ImportError as e:
    if 'torch' not in str(e):
        raise

    TORCH_NOT_FOUND = True


if TORCH_NOT_FOUND and TF_NOT_FOUND:
    raise ModuleNotFoundError('Rinokeras needs PyTorch or Tensorflow to operate')
