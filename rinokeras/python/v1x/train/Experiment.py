from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Callable, Sequence, List

from tensorflow.keras import Model
from tensorflow.contrib.distribute import DistributionStrategy, OneDeviceStrategy

from .train_utils import Inputs, Outputs, Losses


class Experiment(ABC):

    def __init__(self,
                 model: Model,
                 optimizer: str = 'adam',
                 learning_rate: Union[float, Callable[[int], float]] = 1e-3,
                 gradient_clipping: str = 'none',
                 gradient_clipping_bounds: Union[float, Tuple[float, float]] = (-1, 1),
                 return_loss_summaries: bool = False,
                 return_variable_summaries: bool = False,
                 return_grad_summaries: bool = False,
                 distribution_strategy: DistributionStrategy = OneDeviceStrategy('/gpu:0')) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping
        self.gradient_clipping_bounds = gradient_clipping_bounds
        self.return_loss_summaries = return_loss_summaries
        self.return_variable_summaries = return_variable_summaries
        self.return_grad_summaries = return_grad_summaries
        self.distribution_strategy = distribution_strategy

    @abstractmethod
    def build_model(self, inputs: Inputs) -> Outputs:
        return NotImplemented

    @abstractmethod
    def loss_function(self, inputs: Inputs, outputs: Outputs) -> Losses:
        return NotImplemented
