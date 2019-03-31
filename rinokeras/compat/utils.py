"""
Compatability utils
"""
from packaging import version
import sys

if version.parse(sys.version) < version.parse("3.5"):
    from .py34_utils import *
else:
    from .py35_utils import *
