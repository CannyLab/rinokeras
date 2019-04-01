"""
Configure the testing for multiple versions. 

If we're using tensorflow 2.x, we want to only run the tensorflow 2 testing, same
if we are using tensorflow 1.x
"""

import tensorflow as tf
from packaging import version
collect_ignore = ['setup.py']

# Check the tensorflow version
if version.parse(tf.__version__) < version.parse("2.0.0a0"):
    collect_ignore.append("rinokeras/testing/v2x/")    
else:
    collect_ignore.append("rinokeras/testing/v1x/")