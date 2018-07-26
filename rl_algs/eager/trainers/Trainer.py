import tensorflow as tf

class Trainer(object):

    def __init__(self, model, optimizer: str = 'adam') -> None:
        self._model = model

        self._num_param_updates = 0
        if optimizer == 'adam':
            self._optimizer = tf.train.AdamOptimizer()
        elif optimizer == 'rmsprop':
            self._optimizer = tf.train.RMSPropOptimizer()
        else:
            raise ValueError("Unrecognized optimizer. Received {}.".format(optimizer))

        self._inputs_setup = False

    def _batch_norm(self, array, mean, var):
        array = array - mean
        array = array / (tf.sqrt(var) + 1e-10)
        return array

    def loss_function(self, features, labels, *args, **kwargs):
        raise NotImplementedError("Must implement a loss function.")

    def grads_function(self, features, labels, *args, **kwargs):
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                loss = self.loss_function(features, labels, *args, **kwargs)
        
            total_loss, losses = self._unpack_losses(loss)
            grads = tape.gradient(total_loss, self._model.variables)

        else:
            loss = self.loss_function(features, labels, *args, **kwargs)
            total_loss, losses = self._unpack_losses(loss)
            grads = self._optimizer.compute_gradients(total_loss, self._model.variables)

        return grads, losses

    def _unpack_losses(self, losses):
        if isinstance(losses, tuple) or isinstance(losses, list):
            total_loss = losses[0]
            losses = losses
        else:
            total_loss = losses

        return total_loss, losses

    def _train_on_batch(self, features, labels, *args, learning_rate=1e-3, **kwargs):
        if tf.executing_eagerly():
            grads, loss = self.grads_function(features, labels, *args, **kwargs)
            self._optimizer._lr = learning_rate
            self._optimizer.apply_gradients(zip(grads, self._model.variables))
        else:
            if not self._inputs_setup:
                raise RuntimeError("Calling training op without setting up placeholders.")
            sess = tf.get_default_session()
            if sess is None:
                raise RuntimeError("Must be run inside of a session context when in non-eager mode.")
            feed_dict = {}
            if isinstance(self._feature_in, tuple) or isinstance(self._feature_in, list):
                for f_in, f in zip(self._feature_in, features):
                    feed_dict[f_in] = f
            else:
                feed_dict[self._feature_in] = features
            if isinstance(self._labels_in, tuple) or isinstance(self._labels_in, list):
                for l_in, l in zip(self._labels_in, labels):
                    feed_dict[l_in] = l
            else:
                feed_dict[self._labels_in] = labels
            for arg_in, arg in zip(self._args_in, args):
                feed_dict[arg_in] = arg
            for kw in self._kwargs_in:
                feed_dict[self._kwargs_in[kw]] = kwargs[kw]
            _, loss = sess.run([self._update_op, self._loss], feed_dict=feed_dict)
        return loss

    def train(self, batch, learning_rate=1e-3):
        loss = self._train_on_batch(**batch, learning_rate=learning_rate)
        self._num_param_updates += 1
        return loss

    def setup_from_placeholders(self, features, labels, *args, **kwargs):
        loss = self.loss_function(features, labels, *args, **kwargs)
        total_loss, losses = self._unpack_losses(loss)
        update_op = self._optimizer.minimize(total_loss, var_list=self._model.variables)

        self._feature_in = features
        self._labels_in = labels
        self._args_in = args
        self._kwargs_in = kwargs

        self._loss = total_loss
        self._update_op = update_op
        self._inputs_setup = True

    def setup_from_dataset(self, dataset):
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
        batch = iterator.get_next()
        grads, loss = self.loss_function(**batch)
        total_loss, losses = self._unpack_losses(loss)
        update_op = self._optimizer.minimize(total_loss, var_list=self._model.variables)

        self._handle = handle
        self._loss = total_loss
        self._update_op = update_op


    @property
    def num_param_updates(self):
        return self._num_param_updates
