import tensorflow as tf
import os
import sys
import multiprocessing
tf.enable_eager_execution()

from flux.datasets.vision.cifar import Cifar10
from rinokeras.models.resnet import ResNeXt50

# Import Cifar10 Data
cifar = Cifar10()
train_image = tf.convert_to_tensor(cifar.X_train, dtype=tf.float64)
train_labels = tf.convert_to_tensor(cifar.Y_train, dtype=tf.int64)
val_image = tf.convert_to_tensor(cifar.X_test, dtype=tf.float64)
val_labels = tf.convert_to_tensor(cifar.Y_test, dtype=tf.int64)

NUM_EPOCHS = 10000
TEST_INTERVAL = 100
BATCH_SIZE = 4


class PredictionNet(tf.keras.Model):
    def __init__(self, use_layer_norm=True) -> None:
        super(PredictionNet, self).__init__()
        self.resnet = ResNeXt50(use_layer_norm=use_layer_norm)
        self.prediction_module = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=True):
        result = self.resnet(inputs)
        result = self.prediction_module(result)
        # Compute the paddings
        return result

def loss(logits, labels):
    sparse_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    return tf.reduce_mean(sparse_loss)

def compute_accuracy(logits, labels):
    index = tf.argmax(logits, axis=1)
    values = tf.cast(tf.equal(index, labels), tf.float64)

    return tf.reduce_sum(values)/float(BATCH_SIZE)

resnet = PredictionNet(True)
checkpoint_prefix = os.path.join('./checkpoints/', 'ckpt')
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
step_counter = tf.train.get_or_create_global_step()
checkpoint = tf.train.Checkpoint(
    model=resnet, optimizer=optimizer, step_counter=step_counter)
checkpoint.restore(tf.train.latest_checkpoint('./checkpoints/'))

def run():
    for iteration in range(NUM_EPOCHS):
        index = tf.range(0, BATCH_SIZE, delta=1)
        index = tf.random_shuffle(index)

        batch = tf.gather(train_image, index)
        labels = tf.gather(train_labels, index)
        one_hot = tf.one_hot(labels, depth=10, dtype=tf.float64)
        with tf.GradientTape() as tape:
            logits = resnet(batch)
            loss_value = loss(logits, one_hot)
        grads = tape.gradient(loss_value, resnet.variables)
        optimizer.apply_gradients(
            zip(grads, resnet.variables), global_step=step_counter)

        if iteration % 5 == 0:
            print('[Iteration {}] Loss: {}'.format(iteration, loss_value))
            sys.stdout.flush()

        if iteration % TEST_INTERVAL == 0:
            total_accuracy = 0.
            num_batches = 0
            tloss = 0
            index = tf.range(0, BATCH_SIZE, delta=1)
            index = tf.random_shuffle(index)
            batch = tf.gather(val_image, index)
            labels = tf.gather(val_labels, index)
            one_hot = tf.one_hot(labels, depth=10, dtype=tf.float64)
            logits = resnet(batch)
            tloss += loss(logits,one_hot)
            total_accuracy += compute_accuracy(logits, labels)
            num_batches += 1
            print('[TEST ITERATION, Iteration {}] Validation Accuracy: {}, Validation Loss: {}'.format(
                iteration, float(total_accuracy) / num_batches, float(tloss) / num_batches))
            checkpoint.save(checkpoint_prefix)
            sys.stdout.flush()

if __name__ == "__main__":
    run()