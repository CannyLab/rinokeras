

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import warnings
warnings.simplefilter('error', tf.errors.OutOfRangeError)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from rinokeras.core.v2x.train import Experiment

class TestExperiment(Experiment):

    def __init__(self,):
        self.distribution_strategy = tf.distribute.MirroredStrategy()

    def get_model(self,):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        return model

    def get_loss_function(self, ):
        loss = tf.losses.SparseCategoricalCrossentropy()
        def loss_function(model_outputs, inputs):
            c_loss = loss(y_true=inputs[1], y_pred=model_outputs)
            return c_loss, {'loss': c_loss}
        return loss_function

    def get_optimizer(self,):
        return tf.optimizers.Adam()

    def forward(self, model, inputs):
        return model(inputs[0])
    

def test_ExperimentRunner_sequential():

    datasets, ds_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']

    BUFFER_SIZE = 10000
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 2

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE, drop_remainder=True).repeat()

    # Build the experiment
    exp = TestExperiment()
    runner = exp.runner()
    # Train the model
    runner.train(train_dataset, eval_dataset, 1, n_iterations_per_epoch_train=30, n_iterations_per_epoch_eval=30)
        
if __name__ == '__main__':
    test_ExperimentRunner_sequential()
