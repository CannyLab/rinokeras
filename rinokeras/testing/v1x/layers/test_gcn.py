import numpy as np
import tensorflow as tf
import warnings
import tempfile

from rinokeras.testing.utils import *

def test_gcn():
    reset_session()
    # Construct the layer
    from rinokeras.core.v1x.common.gcn import GraphConvolutionalLayer
    layer = GraphConvolutionalLayer(units=8)

    # Make sure that the layer is not None
    assert layer is not None

    # Encoded values
    input_tensor, _ = random_tensor((16, 64, 128))
    adj_matrix, _ = random_tensor((16, 64, 64))

    # Get the output of the layer
    value = layer(input_tensor, adj_matrix)

    # Create a named temporary file for save/restore testing
    with tempfile.TemporaryFile() as weights_file:

        # Construct the session
        output = run_simple_session_save_weights(inputs=[value],
                                                 feed={},
                                                 weights=[layer],
                                                 weights_file=weights_file)

        # Make sure the value is not null
        assert output[0] is not None

        # Make sure the output shape is correct
        assert output[0].shape == (16,64,8)

        # Make sure the output values are correct (If Possible)
        pass

        # Check loading and restoring
        load_restore_test(output=output,
                          inputs=[value],
                          feed={},
                          weights=[layer],
                          weights_file=weights_file)

    # Do regression testing
    check_regression('gcn_layer_expected_output',
                     output, __file__, 'regression_outputs/test_gcn_outputs.json')
