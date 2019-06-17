from typing import Tuple, Dict, Callable
import tensorflow as tf
import numpy as np
import tqdm
import warnings

from rinokeras.core.v2x.train import MetricsAccumulator

class BaseExperimentRunner:
    def __init__(self, experiment) -> None:
        self.experiment = experiment

        with self.experiment.distribution_strategy.scope():

            # Build the model, optimizer and checkpoint
            self._built_model = self.experiment.get_model()
            self._built_optimizer = self.experiment.get_optimizer()
            self._built_loss = self.experiment.get_loss_function()

            self._built_checkpoint = tf.train.Checkpoint(optimizer=self._built_optimizer, model=self._built_model)

            # Make sure that the model is callable
            if not hasattr(self._built_model, '__call__'):
                raise AssertionError('Model must be callable')
            # If model doesn't have trainable variables, add an empty list
            if not hasattr(self._built_model, 'trainable_variables'):
                warnings.warn('Model has no trainable variables - training this model will do nothing.', RuntimeWarning)
                setattr(self._built_model, 'trainable_variables', [])

            # Build the forward pass
            def _built_forward_pass(inputs, return_outputs=False):
                model_outputs = self.experiment.forward(self._built_model, inputs)
                loss, metrics = self._built_loss(model_outputs, inputs)
                if return_outputs:
                    return loss, metrics, model_outputs
                return loss, metrics
            self._built_forward_pass = _built_forward_pass

            # Build the run_iteration function
            def run_train_iteration(inputs):
                with tf.GradientTape() as tape:
                    loss, metrics = self._built_forward_pass(inputs)
                grads = tape.gradient(loss, self._built_model.trainable_variables)

                # Apply the gradients to the variables
                self._built_optimizer.apply_gradients(zip(grads, self._built_model.trainable_variables))
                return loss, metrics

            self._built_run_train_iteration = run_train_iteration

            def run_eval_iteration(inputs):
                loss, metrics = self._built_forward_pass(inputs)
                return loss, metrics
            self._built_run_eval_iteration = run_eval_iteration

    def train(self, train_dataset, eval_dataset, n_epochs, n_iterations_per_epoch_train, n_iterations_per_epoch_eval):

        # Rename strategy for ease of use
        strategy = self.experiment.distribution_strategy

        with strategy.scope():
            train_iterator = strategy.make_dataset_iterator(train_dataset)
            eval_iterator = strategy.make_dataset_iterator(eval_dataset)

        @tf.function
        def distributed_train():
            with strategy.scope():
                loss, metrics = strategy.experimental_run(self._built_run_train_iteration, train_iterator)
                # Reduce the metrics
                reduced_metrics = {}
                for metric_key, value in metrics.items():
                    reduced_metrics[metric_key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, value)
                # Reduce the loss
                reduced_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss)
                return reduced_loss, reduced_metrics

        @tf.function
        def distributed_eval():
            with strategy.scope():
                loss, metrics = strategy.experimental_run(self._built_run_eval_iteration, eval_iterator)
                # Reduce the metrics
                reduced_metrics = {}
                for metric_key, value in metrics.items():
                    reduced_metrics[metric_key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, value)
                # Reduce the loss
                reduced_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss)
                return reduced_loss, reduced_metrics

        self.len_train_dataset = n_iterations_per_epoch_train
        self.len_eval_dataset = n_iterations_per_epoch_eval

        for epoch in range(n_epochs):

            train_iterator.initialize()
            eval_iterator.initialize()

            # Run the training data
            train_losses = []
            train_metrics_acc = MetricsAccumulator()
            train_metrics_acc.start_timer()
            with tqdm.tqdm(total=self.len_train_dataset, leave=False, desc="Epoch {} Train".format(epoch)) as pbar:
                for _ in range(n_iterations_per_epoch_train):
                    loss, train_metrics = distributed_train()
                    train_losses.append(loss.numpy())
                    train_metrics_acc.add(train_metrics)
                    pbar.update(1)
            train_metrics_acc.end_timer()
            train_metrics = train_metrics_acc.get_average()

            # Run the training data
            eval_losses = []
            eval_metrics_acc = MetricsAccumulator()
            eval_metrics_acc.start_timer()
            with tqdm.tqdm(total=self.len_eval_dataset, leave=False, desc="Epoch {} Validation".format(epoch)) as pbar:
                for _ in range(n_iterations_per_epoch_eval):
                    loss, eval_metrics = distributed_eval()
                    eval_losses.append(loss.numpy())
                    eval_metrics_acc.add(eval_metrics)
                    pbar.update(1)
            eval_metrics_acc.end_timer()
            eval_metrics = eval_metrics_acc.get_average()

            print('[Epoch {}] Train Loss: {}, Eval Loss: {}'.format(epoch, np.mean(train_losses), np.mean(eval_losses)))
            print('[Epoch {}] Train Metrics: {}'.format(epoch, train_metrics))
            print('[Epoch {}] Eval Metrics: {}'.format(epoch, eval_metrics))
        
    # Run a forward pass on the network with a set of inputs
    def forward(self, inputs):
        return self._built_forward_pass(inputs, return_outputs=True)

    @property
    def model(self,):
        return self._built_model
    @property
    def loss(self,):
        return self._built_loss
    @property
    def optimizer(self,):
        return self._built_optimizer