
from .Experiment import Experiment
from rinokeras.core.v2x.train import MINEExperimentRunner

class MINEExperiment(Experiment):
    """
    Forward pass for a MINE experiment should return:
    model_outputs, model_hidden_state, model_x_values, true_outputs
    which is a bit more complicated than the base model structure
    """

    def __init__(self, ):
        pass

    def runner(self,):
        return MINEExperimentRunner(self, 1.0, 1.0)
    