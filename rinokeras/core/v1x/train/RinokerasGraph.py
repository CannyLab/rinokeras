from abc import ABC, abstractmethod
from typing import Sequence, Union, Any, Optional, Dict
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm

from rinokeras.core.v1x.utils import MetricsAccumulator
from .train_utils import Inputs


class RinokerasGraph(ABC):

    _num_graphs = 0

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self._name = self.__class__.__name__.lower()
        if self.__class__._num_graphs > 0:
            self._name += '_{}'.format(RinokerasGraph._num_graphs)
        self.__class__._num_graphs += 1

        self.progress_bar = None
        self.descr_offset = 0
        self.epoch_metrics = None
        self.instrument_idx = 0
        self.inputs = ()

    def _map_to_placeholders(self, placeholders, inputs, feed_dict):
        if isinstance(placeholders, tf.Tensor):
            feed_dict[placeholders] = inputs
        elif isinstance(placeholders, list) and isinstance(inputs, list):
            for ph, input_ in zip(placeholders, inputs):
                self._map_to_placeholders(ph, input_, feed_dict)
        elif isinstance(placeholders, tuple) and isinstance(inputs, tuple):
            for ph, input_ in zip(placeholders, inputs):
                self._map_to_placeholders(ph, input_, feed_dict)
        elif isinstance(placeholders, dict) and isinstance(inputs, dict):
            for key, ph in placeholders:
                self._map_to_placeholders(ph, inputs[key], feed_dict)
        else:
            raise ValueError("Type of placeholders and inputs did not match. Received \
                              {} and {}.".format(type(placeholders), type(inputs)))

    def _get_feed_dict(self, inputs: Optional[Inputs]) -> Optional[Dict[tf.placeholder, Any]]:
        if inputs is None:
            return {}

        feed_dict = {}  # type: Dict[tf.placeholder, Any]
        self._map_to_placeholders(self.inputs, inputs, feed_dict)
        return feed_dict

    def _run_tensor(self, ops: Union[tf.Tensor, Sequence[tf.Tensor]], inputs: Optional[Inputs] = None, instrumented: bool = False) -> Any:
        """Runs the network for a specific tensor

        Args:
            ops (Union[tf.Tensor, Sequence[tf.Tensor]]): op or sequence of ops to run
            *args: Positional arguments to the loss function
            **kwargs: Keyword arguments to the loss function

        Returns:
            Result of running ops

        Raises:
            RuntimeError: If not run inside a tf.Session context
        """
        sess = self._get_session()
        feed_dict = self._get_feed_dict(inputs)

        if instrumented:
            self.instrument_idx += 1
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            results = sess.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            trace_file = tf.gfile.Open(name='timeline_{}'.format(self.instrument_idx), mode='a+')
            trace_file.write(tl.generate_chrome_trace_format(show_memory=True))
        else:
            results = sess.run(ops, feed_dict=feed_dict)
        return results

    def add_progress_bar(self, data_len: Optional[int] = None, epoch_num: Optional[int] = None):
        desc = None if epoch_num is None else 'Epoch {:>3}'.format(epoch_num)
        if data_len is None and self.epoch_metrics is not None:
            data_len = self.epoch_metrics.nupdates
        progress_bar = tqdm(total=data_len, desc=desc, leave=False,
                            dynamic_ncols=True, smoothing=0.1)
        progress_bar.__enter__()
        self.progress_bar = progress_bar
        return self

    def update_progress_bar(self, metrics=None, scroll=True, round_metrics=True):
        if metrics is not None and self.epoch_metrics is not None:
            self.epoch_metrics.add(metrics)
        if self.progress_bar is not None:
            self.progress_bar.update()
            if self.epoch_metrics.nupdates > 0:

                epoch_metric_dict = self.epoch_metrics.get_average()

                # Round off the numbers in the dictionary
                if round_metrics:
                    for k, v in epoch_metric_dict.items():
                        if isinstance(v, float):
                            epoch_metric_dict[k] = round(v, 2)

                postfix = str(epoch_metric_dict)
                if scroll and len(postfix) > 60:
                    self.descr_offset += 1
                    if self.descr_offset + 60 >= len(postfix):
                        self.descr_offset = 0
                    self.progress_bar.set_postfix_str(postfix[self.descr_offset:60 + self.descr_offset])
                else:
                    self.progress_bar.set_postfix(epoch_metric_dict)

    def initialize(self):
        return self

    def __enter__(self):
        self.epoch_metrics = MetricsAccumulator()
        self.epoch_metrics.start_timer()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.progress_bar is not None:
            self.progress_bar.__exit__()
        self.progress_bar = None
        self.epoch_metrics.end_timer()
        return exc_type is None or exc_type == tf.errors.OutOfRangeError

    def _get_session(self) -> tf.Session:
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("Must be run inside of a tf.Session context when in non-eager mode.")
        return sess

    @abstractmethod
    def run(self, ops: Union[str, Sequence[tf.Tensor]], inputs: Optional[Inputs] = None) -> Any:
        return NotImplemented

    @property
    def name(self) -> str:
        return self._name

    @property
    def summary_collection(self) -> str:
        return self.name + '_summaries'

    def run_epoch(self,
                  data_len: Optional[int] = None,
                  epoch_num: Optional[int] = None,
                  summary_writer: Optional[tf.summary.FileWriter] = None) -> MetricsAccumulator:
        with self.add_progress_bar(data_len, epoch_num).initialize():
            while True:
                self.run('default')
        return self.epoch_metrics
