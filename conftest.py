"""
Configure the testing for multiple versions. 

If we're using tensorflow 2.x, we want to only run the tensorflow 2 testing, same
if we are using tensorflow 1.x
"""
from packaging import version

collect_ignore = ['setup.py']
collect_ignore.append('rinokeras/testing/v1x/')
collect_ignore.append('rinokeras/testing/v2x/')
collect_ignore.append('rinokeras/testing/util/')
collect_ignore.append('rinokeras/testing/torch/')

try:
    import tensorflow as tf
    # Check the tensorflow version
    if version.parse(tf.__version__) < version.parse('2.0.0a0'):
        collect_ignore.remove('rinokeras/testing/v1x/')
    else:
        collect_ignore.remove('rinokeras/testing/v2x/')
except ImportError:
    pass

try:
    import torch
    collect_ignore.remove('rinokeras/testing/torch/')
except ImportError:
    pass
