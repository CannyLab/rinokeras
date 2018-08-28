
import numpy as np
import tensorflow as tf
import sys


def test_transformer_preembedding():

    # Setup some constants
    SEQ_LEN = 128
    BATCH_SIZE = 16
    N_SYMBOLS_OUT = 512
    EMBED_SIZE = 300
    EPOCHS = 20

    # Import the transformer
    from rinokeras.models.transformer import Transformer

    # Generate some pre-embedded data
    sample_batch_x = np.random.rand(
        BATCH_SIZE, SEQ_LEN, EMBED_SIZE).astype(np.float32)
    sample_batch_y_embedded = np.random.rand(
        BATCH_SIZE, SEQ_LEN, EMBED_SIZE).astype(np.float32)
    sample_batch_y_unembedded = np.random.randint(
        0, N_SYMBOLS_OUT, size=(BATCH_SIZE, SEQ_LEN), dtype=np.int32)

    # Build the transformer
    x_placeholder = tf.placeholder(
        tf.float32, [BATCH_SIZE, SEQ_LEN, EMBED_SIZE])
    y_placeholder = tf.placeholder(
        tf.float32, [BATCH_SIZE, SEQ_LEN, EMBED_SIZE])
    y_unembedded_placeholder = tf.placeholder(tf.int32, [None, None])

    transformer_model = Transformer(
        d_model=EMBED_SIZE, n_heads=3, n_symbols_out=N_SYMBOLS_OUT, use_preembedded_vectors=True)
    transformer_out = transformer_model((x_placeholder, y_placeholder))
    loss_fn = tf.contrib.seq2seq.sequence_loss(
        transformer_out, y_unembedded_placeholder, tf.ones([BATCH_SIZE, SEQ_LEN]))

    # Build the optimizer
    optim = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    grads_and_vars = optim.compute_gradients(loss_fn)
    train_step = optim.apply_gradients(grads_and_vars)

    # Now that the model is built, create a session, and run the model
    with tf.Session() as TF_SESSION:
        TF_SESSION.run(tf.global_variables_initializer())
        feed = {x_placeholder: sample_batch_x,
                y_placeholder: sample_batch_y_embedded,
                y_unembedded_placeholder: sample_batch_y_unembedded,
                }

        for epoch in range(EPOCHS):
            # Check for NaN/Inf Gradients
            grads = TF_SESSION.run([g[0] for g in grads_and_vars], feed_dict=feed)
            for idx, pair in enumerate(grads_and_vars):
                gradient = grads[idx]
                variable = pair[1]
                if (np.any(np.isnan(gradient))):
                    weight_matrix = TF_SESSION.run(variable, feed_dict=feed)
                    print(weight_matrix, file=sys.stderr)
                    print(gradient, file=sys.stderr)
                    assert not np.any(np.isnan(gradient)), "Gradient {} has a NaN: {}".format(idx, variable.name)
                assert not np.any(np.isinf(gradient)), "Gradient {} has an inf: {}".format(idx, variable.name)

            # Compute the result/loss
            result, loss = TF_SESSION.run([transformer_out, loss_fn], feed_dict=feed)

            assert not np.any(np.isnan(result)), "Forward pass has NaN values after {} iterations".format(epoch)
            assert not np.any(np.isnan(loss)), "Loss is NaN after {} iterations".format(epoch)
            assert not np.any(np.isinf(result)), "Forward pass has inf values after {} iterations".format(epoch)
            assert not np.any(np.isinf(loss)), "Loss is inf after {} iterations".format(epoch)

            # Run a training step
            TF_SESSION.run(train_step, feed_dict=feed)

            print(loss)
