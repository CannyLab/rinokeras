# Handle some environmental variables
import os

# Rebuild the regression tests during this run
if os.environ.get('RK_REBUILD_REGRESSION_TESTS', None):
    RK_REBUILD_REGRESSION_TESTS = True
else:
    RK_REBUILD_REGRESSION_TESTS = False
