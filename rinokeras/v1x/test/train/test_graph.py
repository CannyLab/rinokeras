import numpy as np
import tensorflow as tf
import pytest

def gpu_setup(req_gpus):
    from tensorflow.python.client import device_lib
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(get_available_gpus()) < req_gpus:
        pytest.skip('Not enough available GPUs to run this test')


def get_test_data():
    x = np.random.sample((1000,50)).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(x)
    return dataset.batch(16)

def get_test_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(1),
    ])

def do_setup():
    model = get_test_model()
    def build_model(inputs):
        return model(inputs)
    def loss_function(inputs, outputs):
        return outputs ** 2
    data = get_test_data()

    return model, build_model, loss_function, data

def test_test_graph_one_device_construction():
    from rinokeras.v1x.train.TestGraph import TestGraph
    tf.reset_default_graph()
    # Sample setup variables
    model, build_model, loss_function, data = do_setup()
    distribution_strategy = tf.contrib.distribute.OneDeviceStrategy('/cpu:0')
    # Check that the graph constructs
    graph = TestGraph(model=model,
                      build_model=build_model,
                      loss_function=loss_function,
                      inputs=data,
                      distribution_strategy=distribution_strategy)
    assert graph is not None

def test_test_graph_multi_device_construction():
    from rinokeras.v1x.train.TestGraph import TestGraph
    tf.reset_default_graph()
    # Sample setup variables
    model, build_model, loss_function, data = do_setup()
    distribution_strategy = tf.contrib.distribute.MirroredStrategy(['/gpu:0','/gpu:1'])
    # Check that the graph constructs
    graph = TestGraph(model=model,
                      build_model=build_model,
                      loss_function=loss_function,
                      inputs=data,
                      distribution_strategy=distribution_strategy)
    assert graph is not None

def test_test_graph_one_device_run_step():
    gpu_setup(1)
    from rinokeras.v1x.train.TestGraph import TestGraph
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        distribution_strategy = tf.contrib.distribute.OneDeviceStrategy('/gpu:0')

        # Sample setup variables
        with distribution_strategy.scope():
            model, build_model, loss_function, data = do_setup()
        
        # Check that the graph constructs
        graph = TestGraph(model=model,
                        build_model=build_model,
                        loss_function=loss_function,
                        inputs=data,
                        distribution_strategy=distribution_strategy)
        assert graph is not None
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        total_losses = 0
        with graph.add_progress_bar(1000, 0).initialize() as g:
            losses = g.run('default')
            assert losses.keys()
            assert losses['Loss'].shape == (16, 1)
            total_losses += np.mean(losses['Loss'])
        assert total_losses > 0

def test_test_graph_multi_device_run_step():
    gpu_setup(2)
    from rinokeras.v1x.train.TestGraph import TestGraph
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        distribution_strategy = tf.contrib.distribute.MirroredStrategy(['/gpu:0','/gpu:1'])

        # Sample setup variables
        with distribution_strategy.scope():
            model, build_model, loss_function, data = do_setup()
        
        # Check that the graph constructs
        graph = TestGraph(model=model,
                        build_model=build_model,
                        loss_function=loss_function,
                        inputs=data,
                        distribution_strategy=distribution_strategy)
        assert graph is not None
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        total_losses = 0
        with graph.add_progress_bar(1000, 0).initialize() as g:
            losses = g.run('default')
            assert losses.keys()
            assert losses['Loss'].shape == (16, 1), losses['Loss'].shape
            total_losses += np.mean(losses['Loss'])
        assert total_losses > 0

def test_train_graph_one_device_construction():
    from rinokeras.v1x.train.TrainGraph import TrainGraph
    tf.reset_default_graph()

    # Sample setup variables
    model, build_model, loss_function, data = do_setup()
    distribution_strategy = tf.contrib.distribute.OneDeviceStrategy('/cpu:0')
    # Check that the graph constructs
    graph = TrainGraph(model=model,
                      optimizer='sgd',
                      build_model=build_model,
                      loss_function=loss_function,
                      inputs=data,
                      distribution_strategy=distribution_strategy)
    assert graph is not None

def test_train_graph_multi_device_construction():
    gpu_setup(2)
    from rinokeras.v1x.train.TrainGraph import TrainGraph
    tf.reset_default_graph()
    # Sample setup variables
    model, build_model, loss_function, data = do_setup()
    distribution_strategy = tf.contrib.distribute.MirroredStrategy(['/gpu:0','/gpu:1'])
    # Check that the graph constructs
    graph = TrainGraph(model=model,
                      build_model=build_model,
                      optimizer='adam',
                      loss_function=loss_function,
                      inputs=data,
                      distribution_strategy=distribution_strategy)
    assert graph is not None

def test_train_graph_one_device_run_step():
    gpu_setup(1)
    from rinokeras.v1x.train.TrainGraph import TrainGraph
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        distribution_strategy = tf.contrib.distribute.OneDeviceStrategy('/gpu:0')

        # Sample setup variables
        with distribution_strategy.scope():
            model, build_model, loss_function, data = do_setup()
        
        # Check that the graph constructs
        graph = TrainGraph(model=model,
                        build_model=build_model,
                        loss_function=loss_function,
                        optimizer='adam',
                        inputs=data,
                        distribution_strategy=distribution_strategy)
        assert graph is not None
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        total_losses = 0
        with graph.add_progress_bar(1000, 0).initialize() as g:
            losses = g.run('default')
            assert losses.keys()
            assert losses['Loss'].shape == (16, 1)
            total_losses += np.mean(losses['Loss'])
        assert total_losses > 0

def test_train_graph_multi_device_run_step():
    gpu_setup(2)
    from rinokeras.v1x.train.TrainGraph import TrainGraph
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        distribution_strategy = tf.contrib.distribute.MirroredStrategy(['/gpu:0','/gpu:1'])

        # Sample setup variables
        with distribution_strategy.scope():
            model, build_model, loss_function, data = do_setup()
        
        # Check that the graph constructs
        graph = TrainGraph(model=model,
                        build_model=build_model,
                        loss_function=loss_function,
                        optimizer='adam',
                        inputs=data,
                        distribution_strategy=distribution_strategy)
        assert graph is not None
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        total_losses = 0
        with graph.add_progress_bar(1000, 0).initialize() as g:
            losses = g.run('default')
            assert losses.keys()
            assert losses['Loss'].shape == (16, 1), losses['Loss'].shape
            total_losses += np.mean(losses['Loss'])
        assert total_losses > 0

def test_train_graph_multi_device_run_multi_step():
    gpu_setup(2)
    from rinokeras.v1x.train.TrainGraph import TrainGraph
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        distribution_strategy = tf.contrib.distribute.MirroredStrategy(['/gpu:0','/gpu:1'])

        # Sample setup variables
        with distribution_strategy.scope():
            model, build_model, loss_function, data = do_setup()
        
        # Check that the graph constructs
        graph = TrainGraph(model=model,
                        build_model=build_model,
                        loss_function=loss_function,
                        optimizer='adam',
                        inputs=data,
                        distribution_strategy=distribution_strategy)
        assert graph is not None
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        total_losses = 0
        with graph.add_progress_bar(1000, 0).initialize() as g:
            losses = g.run('default')
            assert losses.keys()
            assert losses['Loss'].shape == (16, 1), losses['Loss'].shape
            total_losses += np.mean(losses['Loss'])
        
        total_loss_itr_2 = 0
        with graph.add_progress_bar(1000, 0).initialize() as g:
            losses = g.run('default')
            assert losses.keys()
            assert losses['Loss'].shape == (16, 1), losses['Loss'].shape
            total_loss_itr_2 += np.mean(losses['Loss'])

        assert total_losses > total_loss_itr_2