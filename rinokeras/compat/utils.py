"""
Compatability utils
"""
from packaging import version
import sys

def merge_dicts(x, y):
    if version.parse(sys.version) < version.parse("3.5"):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z 
    return {**x, **y}
