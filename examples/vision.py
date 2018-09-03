import tensorflow as tf

from rinokeras.common.layers import Conv2DStack, DenseStack
from rinokeras.trainers import SupervisedTrainer

class ImageClassifier(tf.keras.Model):

    def __init__(self, n_classes: int):
        super(ImageClassifier, self).__init__()
        self.convstack = Conv2DStack(filters=(32, 64, 128), 
                                     kernel_size=(8, 4, 3), 
                                     strides=(4, 2, 1),
                                     activation='relu',
                                     padding='same',
                                     flatten_output=True)
        self.densestack = DenseStack(layers=(300, n_classes))

    def call(self, inputs):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, -1)
        filters = self.convstack(inputs)
        output = self.densestack(filters)
        return output

def run_iteration(trainer, x_batch, y_batch, istraining):
    xbatch = tf.cast(xbatch, tf.float32)
    ybatch = tf.cast(ybatch, tf.int32)

    loss = trainer.train(xbatch, ybatch) if istraining else trainer.loss(xbatch, ybatch)

    return loss.numpy()

if __name__ == '__main__':
    tf.enable_eager_execution()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_data = train_data.shuffle(1000)
    train_data = train_data.batch(64)
    train_data = train_data.prefetch(2)
    train_data = train_data.repeat()

    test_data = test_data.batch(64)

    model = ImageClassifier(10)
    trainer = SupervisedTrainer(model, loss_type='sparse_categorical_crossentropy')

    moving_average = float('inf')
    itr = 0
    while moving_average > 0.01:
        x_batch, y_batch = train_data.get_next()
        loss = run_iteration(trainer, x_batch, y_batch, istraining=True)
        if itr == 0:
            moving_average = loss
        else:
            moving_average = 0.99 * moving_average + 0.01 * loss
        if itf % 10 == 0:
            print("Loss:", moving_average)

    moving_average = 0
    itr = 0
    for xbatch, ybatch in test_data:
        moving_average += run_iteration(trainer, x_batch, y_batch, istraining=False)
        itr += 1

    print("Test Loss:", moving_average / itr)

