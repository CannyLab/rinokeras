
from rinokeras.core.v2x.train import Experiment, MINEExperimentRunner

class MINEExperiment(Experiment):

    def __init__(self, ):
        pass

    def runner(self,):
        return MINEExperimentRunner(self, 1.0, 1.0)
    