import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.contrib.distribute import OneDeviceStrategy
import rinokeras as rk


class SupervisedExperiment(rk.train.Experiment):

    def build_model(self, inputs):
        x, y = inputs
        return self.model(x)

    def loss_function(self, inputs, outputs):
        x, y = inputs
        return tf.losses.mean_squared_error(y, outputs)


x = np.random.random((3000, 128))
y = np.random.random((3000, 1))

data = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)

model = Dense(1)

distribution_strategy = OneDeviceStrategy('/cpu:0')
experiment = SupervisedExperiment(model, distribution_strategy=distribution_strategy)

train_graph = rk.train.TrainGraph.from_experiment(experiment, data)

sess = tf.get_default_session()
if sess is None:
    sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
