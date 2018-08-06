
import numpy as np
import tensorflow as tf
import sys
import os

from flux.datasets.nlp.squad import Squad  # Get the Squad dataset
from flux.processing.nlp.embedding.glove import GloveEmbedding
from rinokeras.models.qanet import QANet  # Get the QANet keras model

# Enable eager execution in tensorflow
tf.enable_eager_execution()

NUM_ITERATIONS = 200000
PRINT_INTERVAL = 1
TEST_INTERVAL = 1000
BATCH_SIZE = 8
VAL_ITERS = 50
MAX_WORD_LEN = 766

# Construct the dataset
dataset = Squad(nohashcheck=True)
print(dataset.info())

# construct the networks


class PredictionNet(tf.keras.Model):
    def __init__(self, word_embed_matrix: np.ndarray, char_embed_matrix: np.ndarray, num_choices: int) -> None:
        super(PredictionNet, self).__init__()
        self.encoder_module = QANet(
            word_embed_matrix=word_embed_matrix, char_embed_matrix=char_embed_matrix)
        self.prediction_module = tf.keras.layers.Dense(units=num_choices)

    def call(self, inputs, training=True):
        result = self.encoder_module(inputs, None, True, training)
        result = self.prediction_module(result)
        # Compute the paddings
        paddings = tf.constant([[0, 0], [0, MAX_WORD_LEN - result.shape[1]], [0, 0]])
        result = tf.pad(result, paddings)
        return result


def loss(logits, labels_st, labels_et):
    start_token_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits[:, :, 0], [BATCH_SIZE, MAX_WORD_LEN]), labels=labels_st))
    end_token_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits[:, :, 1], [BATCH_SIZE, MAX_WORD_LEN]), labels=labels_et))
    return start_token_loss + end_token_loss


def compute_accuracy(logits, labels_st, labels_et):
    s_predictions = tf.argmax(tf.reshape(
        logits[:, :, 0], [BATCH_SIZE, MAX_WORD_LEN]), axis=1, output_type=tf.int64)
    e_predictions = tf.argmax(tf.reshape(
        logits[:, :, 1], [BATCH_SIZE, MAX_WORD_LEN]), axis=1, output_type=tf.int64)
    labels_st = tf.cast(labels_st, tf.int64)
    labels_et = tf.cast(labels_et, tf.int64)
    batch_size = int(logits.shape[0])
    return (tf.reduce_sum(tf.cast(tf.equal(s_predictions, labels_st), dtype=tf.float32)) + tf.reduce_sum(tf.cast(tf.equal(e_predictions, labels_et), dtype=tf.float32))) / (2 * batch_size)


# Build the embedding matrices
word_embedding_matrix = GloveEmbedding().GenerateMatrix(dataset.dictionary.word_dictionary)  # np.random.random(size=(dataset.word_vocab_size, 300))
char_embedding_matrix = np.random.random(size=(dataset.char_vocab_size, 200))

print(word_embedding_matrix.shape, char_embedding_matrix.shape)

# Build the model
model = PredictionNet(word_embed_matrix=word_embedding_matrix,
                      char_embed_matrix=char_embedding_matrix, num_choices=2)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

checkpoint_prefix = os.path.join('./checkpoints/', 'ckpt')
step_counter = tf.train.get_or_create_global_step()
checkpoint = tf.train.Checkpoint(
    model=model, optimizer=optimizer, step_counter=step_counter)
checkpoint.restore(tf.train.latest_checkpoint('./checkpoints/'))

# Build the DB for the records
train_db = dataset.train_db.shuffle(
    buffer_size=3000).repeat().batch(BATCH_SIZE)
val_db = dataset.val_db.shuffle(buffer_size=3000).repeat().batch(BATCH_SIZE)
train_iterator = tf.contrib.eager.Iterator(train_db)
val_iterator = tf.contrib.eager.Iterator(val_db)

# Train the model
for iteration in range(NUM_ITERATIONS):

    batch = next(train_iterator)

    with tf.GradientTape() as tape:
        logits = model(batch, training=True)
        loss_value = loss(logits, batch[4], batch[5])
    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(
        zip(grads, model.variables), global_step=step_counter)

    if iteration % 5 == 0:
        print('[Iteration {}] Loss: {}'.format(iteration, loss_value))
        sys.stdout.flush()

    if iteration % TEST_INTERVAL == 0:
        total_accuracy = 0.
        num_batches = 0
        tloss = 0
        for vi in range(VAL_ITERS):
            vbatch = next(val_iterator)
            logits = model(vbatch)
            tloss += loss(logits, vbatch[4], vbatch[5])
            total_accuracy += compute_accuracy(logits, vbatch[4], vbatch[5])
            num_batches += 1
        print('[TEST ITERATION, Iteration {}] Validation Accuracy: {}, Validation Loss: {}'.format(
            iteration, float(total_accuracy) / num_batches, float(tloss) / num_batches))
        checkpoint.save(checkpoint_prefix)
        sys.stdout.flush()
