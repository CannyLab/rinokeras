"""
Utilities for regression testing
"""
import json
import os
import warnings
import numpy as np
import tensorflow as tf

def get_local_file(fpath, file__):
   return '/'+os.path.join(os.path.join(*file__.split(os.sep)[:-1]), fpath)

def check_regression(regression_key, output, file__, fname, debug=False, tol=1e-3):
    regression_key = str(tf.__version__) + '_' + regression_key
    try:
        with open(get_local_file(fname, file__), 'r') as json_file:
            jf = json.loads(json_file.read())
    except FileNotFoundError:
        warnings.warn('{} not found. Creating it.'.format(fname))
        jf = {}
    if not debug and regression_key in jf:
        with open(get_local_file(fname, file__), 'r') as json_file:
            jf = json.loads(json_file.read())
            expected_output = [np.array(v) for v in jf[regression_key]]
    else:
        if isinstance(output, (list, tuple)):
            jf[regression_key] = [i.tolist() for i in output]
            expected_output = output
        else:
            jf[regression_key] = [output.tolist()]
            expected_output = [output]
        with open(get_local_file(fname, file__), 'w') as json_file:
            json.dump(jf, json_file)
        warnings.warn('Regression test not found for {} in {}: Building this\
             now.'.format(regression_key, fname))

    # Now do assertions
    if not isinstance(output, (list, tuple)):
        output = [output]
    for x, y in zip(output, expected_output):
        assert np.isclose(x, y, rtol=tol, atol=tol).all()
