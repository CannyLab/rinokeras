"""
Compatability utils
"""
from packaging import version
import sys
from rinokeras import RK_USE_TF_VERSION

def tf2():
    return RK_USE_TF_VERSION == 2

if version.parse(sys.version) < version.parse("3.5"):
    from .py34_utils import *
else:
    from .py35_utils import *
