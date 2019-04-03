

from abc import ABC, abstractmethod
from rinokeras.core.v2x.train import BaseExperimentRunner

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
    



