from typing import Optional

import tensorflow as tf

from rinokeras.common.layers import Conv2DStack, DenseStack, PositionEmbedding2D
from rinokeras.models.transformer import TransformerEncoder
from rinokeras.trainers import SupervisedTrainer

class ImageClassifier(tf.keras.Model):

    def __init__(self, n_classes: int) -> None:
        super(ImageClassifier, self).__init__()
        self.convstack = Conv2DStack(filters=(32, 64, 128), 
                                     kernel_size=(8, 4, 3), 
                                     strides=(4, 2, 1),
                                     activation='relu',
                                     padding='same',
                                     flatten_output=True)
        self.densestack = DenseStack(layers=(300, n_classes))

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
                 n_layers: int,
                 n_heads: int,
                 dropout: Optional[float] = None) -> None:
        super(ImageTransformer, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_model = d_model = 128
        self.d_filter = d_filter = 4 * d_model

        self.convstack = Conv2DStack(filters=(32, 64, d_model),
                                     kernel_size=(8, 4, 3),
                                     strides=(4, 2, 1),
                                     activation='relu',
                                     padding='same',
                                     flatten_output=False)
        self.pos_embed_2d = PositionEmbedding2D()
        self.transformer = TransformerEncoder(
            n_layers, n_heads, d_model, d_filter, dropout)

        self.flatten = tf.keras.layers.Flatten()
        self.densestack = DenseStack(layers=(300, n_classes))

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

        batch_size, new_width, new_height = (tf.shape(embedded)[i] for i in range(3))

        # transformer_input: Tensor with shape [batch_size, width / something * height / something, self.d_model]
        transformer_input = tf.reshape(embedded, (batch_size, new_width * new_height, self.d_model))

        # transformer_output: Tensor with shape [batch_size, width / something * height / something, self.d_model]
        transformer_output = self.transformer(transformer_input, encoder_mask=None)

        # logits: Tensor with shape [batch_size, width / something * height / something * self.d_model)]
        transformer_output_flat = self.flatten(transformer_output)
        output = self.densestack(transformer_output_flat)

        return output


def run_iteration(trainer, x_batch, y_batch, istraining):
    x_batch = tf.cast(x_batch, tf.float32)
    y_batch = tf.cast(y_batch, tf.int32)

    loss = trainer.train(x_batch, y_batch) if istraining else trainer.loss(x_batch, y_batch)

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
        if itr % 10 == 0:
            print("Loss:", moving_average)

    moving_average = 0
    itr = 0
    for xbatch, ybatch in test_data:
        moving_average += run_iteration(trainer, x_batch, y_batch, istraining=False)
        itr += 1

    print("Test Loss:", moving_average / itr)
