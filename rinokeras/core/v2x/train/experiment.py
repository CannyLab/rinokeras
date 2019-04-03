
from typing import Tuple, Dict, Callable
import tensorflow as tf
from abc import ABC, abstractmethod
import numpy as np
import tqdm

class Experiment(ABC):

    def __init__(self, ):
        pass

    @abstractmethod
    def get_model(self,):
        raise NotImplementedError('Experiment must override the get model function')

    @abstractmethod
    def get_optimizer(self,):
        raise NotImplementedError('Experiment must override the get optimizer function')

    @abstractmethod
    def get_loss_function(self, ):
        raise NotImplementedError('Experiment must override get loss function')

    @abstractmethod
    def forward(self, model, inputs):
        raise NotImplementedError('Experiment must override forward function')    

    def runner(self,):
        return BaseExperimentRunner(self)
    

class BaseExperimentRunner:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

        with self.experiment.distribution_strategy.scope():

            # Build the model, optimizer and checkpoint
            self._built_model = self.experiment.get_model()
            self._built_optimizer = self.experiment.get_optimizer()
            self._built_loss = self.experiment.get_loss_function()
            self._built_checkpoint = tf.train.Checkpoint(optimizer=self._built_optimizer, model=self._built_model)

            # Build the forward pass
            def _built_forward_pass(inputs):
                model_outputs = self.experiment.forward(self._built_model, inputs)
                loss, metrics = self._built_loss(model_outputs, inputs)
                return loss, metrics
            self._built_forward_pass = _built_forward_pass

            # Build the run_iteration function
            def run_train_iteration(inputs):
                with tf.GradientTape() as tape:
                    loss, metrics = self._built_forward_pass(inputs)
                grads = tape.gradient(loss, self._built_model.trainable_variables)

                # Apply the gradients to the variables
                self._built_optimizer.apply_gradients(zip(grads, self._built_model.trainable_variables))
                return loss

            self._built_run_train_iteration = run_train_iteration

            def run_eval_iteration(inputs):
                loss, metrics = self._built_forward_pass(inputs)
                return loss
            self._built_run_eval_iteration = run_eval_iteration

    def train(self, train_dataset, eval_dataset, n_epochs, n_iterations_per_epoch_train, n_iterations_per_epoch_eval):

        with self.experiment.distribution_strategy.scope():
            train_iterator = self.experiment.distribution_strategy.make_dataset_iterator(train_dataset)
            eval_iterator = self.experiment.distribution_strategy.make_dataset_iterator(eval_dataset)

        @tf.function
        def distributed_train():
            with self.experiment.distribution_strategy.scope():
                result = self.experiment.distribution_strategy.experimental_run(self._built_run_train_iteration, train_iterator)
                return self.experiment.distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, result)

        @tf.function
        def distributed_eval():
            with self.experiment.distribution_strategy.scope():
                result = self.experiment.distribution_strategy.experimental_run(self._built_run_eval_iteration, eval_iterator)
                return self.experiment.distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, result)

        self.len_train_dataset = n_iterations_per_epoch_train
        self.len_eval_dataset = n_iterations_per_epoch_eval

        for epoch in range(n_epochs):

            train_iterator.initialize()
            eval_iterator.initialize()

            # Run the training data
            train_losses = []
            with tqdm.tqdm(total=self.len_train_dataset, leave=False, desc="Epoch {} Train".format(epoch)) as pbar:
                for idx in range(n_iterations_per_epoch_train):
                    loss = distributed_train()
                    train_losses.append(loss.numpy())
                    pbar.update(1)

            # Run the training data
            valid_losses = []
            with tqdm.tqdm(total=self.len_eval_dataset, leave=False, desc="Epoch {} Validation".format(epoch)) as pbar:
                for idx in range(n_iterations_per_epoch_eval):
                    loss = distributed_eval()
                    valid_losses.append(loss.numpy())
                    pbar.update(1)

            print('[Epoch {}] Train Loss: {}, Valid Loss: {}'.format(epoch, np.mean(train_losses), np.mean(valid_losses)))
        

    # def evaluate(self, inputs):
    #     return self.experiment.model(inputs)

