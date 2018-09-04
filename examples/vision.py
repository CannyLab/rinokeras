from typing import Optional
import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from rinokeras.common.layers import Conv2DStack, DenseStack, PositionEmbedding2D
from rinokeras.models.transformer import TransformerEncoder
from rinokeras.trainers import SupervisedTrainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class ImageClassifier(tf.keras.Model):

    def __init__(self, n_classes: int) -> None:
        super(ImageClassifier, self).__init__()
        self.convstack = Conv2DStack(filters=(32, 64, 128),
                                     kernel_size=(5, 3, 3),
                                     strides=(2, 1, 1),
                                     activation='relu',
                                     padding='same',
                                     flatten_output=True)
        self.densestack = DenseStack(layers=(512, n_classes))

    def call(self, inputs):
        """
        Args:
            inputs: a uint8/float32 Tensor with shape [batch_size, width, height, channels]

        Returns:
            output: a float32 Tensor with shape [batch_size, n_classes]
        """
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, -1)
        filters = self.convstack(inputs)
        output = self.densestack(filters)

        return output


class ImageTransformer(tf.keras.Model):

    def __init__(self,
                 n_classes: int,
                 n_layers: int = 3,
                 n_heads: int = 4,
                 dropout: Optional[float] = None) -> None:
        super(ImageTransformer, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_model = d_model = 128
        self.d_filter = d_filter = 4 * d_model

        self.convstack = Conv2DStack(filters=(32, 64, d_model),
                                     kernel_size=(5, 3, 3),
                                     strides=(2, 1, 1),
                                     activation='relu',
                                     padding='same',
                                     flatten_output=False)
        self.pos_embed_2d = PositionEmbedding2D()
        self.transformer = TransformerEncoder(
            n_layers, n_heads, d_model, d_filter, dropout)

        self.flatten = tf.keras.layers.Flatten()
        self.densestack = DenseStack(layers=(512, n_classes))

    def call(self, inputs):
        """
        Args:
            inputs: a uint8/float32 Tensor with shape [batch_size, width, height, channels]

        Returns:
            output: a float32 Tensor with shape [batch_size, n_classes]
        """
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, -1)

        # filters: Tensor with shape [batch_size, width / something, height / something, self.d_model]
        filters = self.convstack(inputs)
        # embedded: Tensor with shape [batch_size, width / something, height / something, self.d_model]
        embedded = self.pos_embed_2d(filters)
        batch_size = tf.shape(embedded)[0]
        new_width, new_height = embedded.shape[1], embedded.shape[2]

        # transformer_input: Tensor with shape [batch_size, width / something * height / something, self.d_model]
        transformer_input = tf.reshape(embedded, (batch_size, new_width * new_height, self.d_model))

        # transformer_output: Tensor with shape [batch_size, width / something * height / something, self.d_model]
        transformer_output = self.transformer(transformer_input, encoder_mask=None)
        transformer_output = tf.nn.relu(transformer_output)
        # logits: Tensor with shape [batch_size, width / something * height / something * self.d_model)]
        transformer_output_flat = self.flatten(transformer_output)
        # transformer_output_flat = self.flatten(embedded)
        output = self.densestack(transformer_output_flat)

        return output


def run_iteration(trainer, x_batch, y_batch, istraining):
    x_batch = tf.cast(x_batch, tf.float32)
    y_batch = tf.cast(y_batch, tf.int32)

    loss = trainer.train(x_batch, y_batch) if istraining else trainer.loss(x_batch, y_batch)

    return loss.numpy()


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    x_test = np.transpose(x_test, (0, 2, 3, 1))

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def cast_data(x, y):
        return tf.cast(x, tf.float32), tf.cast(y, tf.int32)

    train_data = train_data.map(cast_data)
    test_data = test_data.map(cast_data)

    train_data = train_data.shuffle(1000)
    train_data = train_data.batch(64)
    train_data = train_data.prefetch(2)

    test_data = test_data.batch(64)

    model = ImageTransformer(100)
    trainer = SupervisedTrainer(model=model, loss_type='sparse_categorical_crossentropy', optimizer='sgd', gradient_clipping='value')
    trainer.setup_from_dataset(train_data)

    sess.run(tf.global_variables_initializer())

    data_len = 1000

    best_accuracy = float('-inf')

    for epoch in range(500):
        train_iter = train_data.make_one_shot_iterator()
        train_handle = sess.run(train_iter.string_handle())
        test_iter = test_data.make_one_shot_iterator()
        test_handle = sess.run(test_iter.string_handle())
        loss = 0.
        n_minibatches = 0.
        with tqdm(total=data_len, desc='Epoch {:>3}'.format(epoch), leave=False, dynamic_ncols=True, smoothing=0) as progress_bar:
            try:
                while True:
                    loss += trainer.train(train_handle)
                    n_minibatches += 1
                    progress_bar.update()
                    progress_bar.set_postfix(Loss=float(loss / n_minibatches))
            except tf.errors.OutOfRangeError:
                loss = float(loss) / n_minibatches
                data_len = n_minibatches

        test_acc = 0.
        test_minibatches = 0.
        try:
            while True:
                curr_acc = sess.run(trainer.accuracy, feed_dict={trainer._handle: test_handle})
                test_acc += curr_acc
                # print(pred)
                # test_loss += trainer.loss(test_handle)
                test_minibatches += 1.
        except tf.errors.OutOfRangeError:
            test_acc /= test_minibatches

        best_accuracy = max(best_accuracy, test_acc)

        print('Epoch {:>3}: Loss {:0.3f}, Test Accuracy {:0.2%}, Best Accuracy {:0.2%}'.format(epoch, loss, test_acc, best_accuracy))
