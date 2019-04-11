
import tensorflow as tf
import warnings
from .BaseExperimentRunner import BaseExperimentRunner

class MINEEstimator(tf.keras.Model):
    def __init__(self, ):
        super(MINEEstimator, self).__init__()
        self._dense_1 = tf.keras.layers.Dense(self.hidden_size)
        self._dense_2 = tf.keras.layers.Dense(self.hidden_size)
        self._predict_MI = tf.keras.layers.Dense(1)

    def __call__(self, inputs, mask=None, **kwargs):
        x_values, y_values = inputs
        x_compressed = self._dense_1(x_values)
        y_compressed = self._dense_2(y_values)
        h1 = tf.nn.relu(x_compressed + y_compressed)
        h2 = tf.nn.relu(x_compressed + tf.random.shuffle(y_compressed))
        estimate = tf.reduce_mean(self._predict_MI(h1)) - tf.math.log(tf.reduce_mean(tf.math.exp(self._predic_MI(h2))))
        return estimate

class MINEExperimentRunner(BaseExperimentRunner):

    def __init__(self, experiment, gamma, beta):
        self.experiment = experiment

        # Training parameters for loss tradeoffs
        self._gamma = gamma
        self._beta = beta

        with self.experiment.distribution_strategy.scope():

            # Build the model, optimizer and checkpoint
            self._built_model = self.experiment.get_model()
            self._built_optimizer = self.experiment.get_optimizer()
            self._built_loss = self.experiment.get_loss_function()

            self._built_checkpoint = tf.train.Checkpoint(optimizer=self._built_optimizer, model=self._built_model)

            # Build the MINE estimator
            self._mine_xz_estimator = MINEEstimator()
            self._mine_xy_estimator = MINEEstimator()
            self._mi_optimizer = tf.optimizers.Adam()

            # Make sure that the model is callable
            if not hasattr(self._built_model, '__call__'):
                raise AssertionError('Model must be callable')
            # If model doesn't have trainable variables, add an empty list
            if not hasattr(self._built_model, 'trainable_variables'):
                warnings.warn('Model has no trainable variables - training this model will do nothing.', RuntimeWarning)
                setattr(self._built_model, 'trainable_variables', [])

            # Build the forward pass
            def _built_forward_pass(inputs, return_outputs=False):
                model_outputs, model_hidden_state, model_x_values, true_outputs = self.experiment.forward(self._built_model, inputs)

                xz_mi_estimate = self._mine_xz_estimator(model_x_values, model_hidden_state)
                yz_mi_estimate = self._mine_xy_estimator(model_hidden_state, true_outputs)

                # Optimize the MI Estimates (We want to Max I(Z,Y))
                loss, metrics = self._built_loss(model_outputs, inputs) + self._gamma * xz_mi_estimate - self._beta * yz_mi_estimate
                if return_outputs:
                    return (loss, xz_mi_estimate, yz_mi_estimate), metrics, model_outputs
                return (loss, xz_mi_estimate, yz_mi_estimate), metrics
            self._built_forward_pass = _built_forward_pass

            # Build the run_iteration function
            def run_train_iteration(inputs):
                with tf.GradientTape(persistent=True) as tape:
                    loss, metrics = self._built_forward_pass(inputs)

                model_loss, xz_mi_estimate, yz_mi_estimate = loss
                
                # Get the mutual information gradients
                xz_mi_gradients = tape.gradient(-xz_mi_estimate, self._mine_xz_estimator.trainable_variables) # Maximize the VIB
                yz_mi_gradients = tape.gradient(-yz_mi_estimate, self._mine_yz_estimator.trainable_variables) # Maximize the VIB

                # Get the gradients for the actual model
                grads = tape.gradient(model_loss, self._built_model.trainable_variables)

                # Apply the gradients to the variables
                self._mi_optimizer.apply_gradients(zip(xz_mi_gradients, self._mine_xz_estimator.trainable_variables))
                self._mi_optimizer.apply_gradients(zip(yz_mi_gradients, self._mine_yz_estimator.trainable_variables))
                
                self._built_optimizer.apply_gradients(zip(grads, self._built_model.trainable_variables))
                return loss, metrics

            self._built_run_train_iteration = run_train_iteration

            def run_eval_iteration(inputs):
                loss, metrics = self._built_forward_pass(inputs)
                model_loss, xz_mi_estimate, yz_mi_estimate = loss
                metrics['XZ_MI'] = xz_mi_estimate
                metrics['XZ_MI'] = yz_mi_estimate
                return model_loss, metrics
            self._built_run_eval_iteration = run_eval_iteration
