
import numpy as np
import tensorflow as tf

from flux.datasets.nlp.newslens import NLQA  # Get the NLQA dataset
from rinokeras.models.qanet import QANet  # Get the QANet keras model

# Enable eager execution in tensorflow
tf.enable_eager_execution()

NUM_ITERATIONS = 50000
PRINT_INTERVAL = 10
TEST_INTERVAL = 1000
BATCH_SIZE = 16

# Construct the dataset
dataset = NLQA()


# construct the networks
class PredictionNet(tf.keras.Model):
    def __init__(self, word_embed_matrix: np.ndarray, char_embed_matrix: np.ndarray, num_choices: int) -> None:
        super(PredictionNet, self).__init__()
        self.encoder_module = QANet(word_embed_matrix=word_embed_matrix, char_embed_matrix=char_embed_matrix)
        self.prediction_module = tf.keras.layers.Dense(units=num_choices)

    def call(self, inputs, training=True):
        result = tf.reshape(self.encoder_module(inputs, None, True, training), (BATCH_SIZE, -1))
        result = self.prediction_module(result)
        return result


def loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


# Build the embedding matrices
word_embedding_matrix = np.random.random(size=(dataset.word_vocab_size,300))
char_embedding_matrix = np.random.random(size=(dataset.char_vocab_size,200))

print(word_embedding_matrix.shape, char_embedding_matrix.shape)

# Build the model
model = PredictionNet(word_embed_matrix=word_embedding_matrix, char_embed_matrix=char_embedding_matrix,num_choices=dataset.multiple_choice_max_size)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)

# Train the model
for iteration in range(NUM_ITERATIONS):

    feature_data, labels = dataset.sample_qanet_train(size=BATCH_SIZE)
    feature_data = (
        tf.convert_to_tensor(feature_data[0]),
        tf.convert_to_tensor(feature_data[1]),
        tf.convert_to_tensor(feature_data[2]),
        tf.convert_to_tensor(feature_data[3]),
        tf.convert_to_tensor(feature_data[4]),
        tf.convert_to_tensor(feature_data[5]),
        None)
    labels = tf.convert_to_tensor(labels)
    # feature_data = (Context, Question, Context chars, Question chars, a1, a2)

    with tf.GradientTape() as tape:
        logits = model(feature_data, training=True)
        loss_value = loss(logits, labels)
    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))

    if iteration % 1 == 0:
        print('[Iteration {}] Loss: {}'.format(iteration, loss_value))

    if iteration % TEST_INTERVAL == 10:
        total_accuracy = 0.
        num_batches = 0
        for (feature_data, labels) in dataset.iterate_validation_set(batch_size=BATCH_SIZE):
            feature_data = (
                tf.convert_to_tensor(feature_data[0]),
                tf.convert_to_tensor(feature_data[1]),
                tf.convert_to_tensor(feature_data[2]),
                tf.convert_to_tensor(feature_data[3]),
                tf.convert_to_tensor(feature_data[4]),
                tf.convert_to_tensor(feature_data[5]),
                None)
            total_accuracy += compute_accuracy(model(feature_data), labels)
            num_batches += 1
        print('[TEST ITERATION, Iteration {}] Validation Accuracy: {}'.format(iteration, float(total_accuracy) / num_batches))




# Context
# Question
# Context characters
# Question characters
# Answer index 1
# Answer index 2