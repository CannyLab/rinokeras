import tensorflow as tf
from packaging import version

RK_USE_TF_VERSION = None
if version.parse(tf.__version__) < version.parse("2.0.0"):
    RK_USE_TF_VERSION = 1
else:
    RK_USE_TF_VERSION = 2