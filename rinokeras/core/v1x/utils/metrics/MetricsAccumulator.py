import warnings
from timeit import default_timer as timer
from typing import Dict
from collections import defaultdict


class MetricsAccumulator:

    def __init__(self):
        self._totalmetrics = defaultdict(lambda: 0.0)
        self._nupdates = 0
        self._start_time = float('nan')

    def add(self, metrics: Dict[str, float]):
        for metric, value in metrics.items():
            self._totalmetrics[metric] += value
        self._nupdates += 1

    def start_timer(self):
        self._start_time = timer()

    def end_timer(self):
        self.runtime = timer() - self._start_time

    def get_average(self):
        assert self.nupdates > 0
        return {metric: value / self._totalmetrics.get('batch_size', self.nupdates)
                for metric, value in self._totalmetrics.items() if metric != 'batch_size'}

    def __iter__(self):
        return iter(self.get_average())

    def items(self):
        return self.get_average().items()

    def __getitem__(self, key: str) -> float:
        if key not in self._totalmetrics:
            raise KeyError(key)
        if key == 'batch_size':
            warnings.warn('Getting the batch size from this is pretty weird')
        return self._totalmetrics[key] / self._totalmetrics.get('batch_size', self.nupdates)

    def __str__(self) -> str:
        return str(self.get_average())

    @property
    def nupdates(self) -> int:
        return self._nupdates
