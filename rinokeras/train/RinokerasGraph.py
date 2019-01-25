from abc import ABC, abstractmethod
from typing import Sequence, Union, Any, Optional, Dict

import tensorflow as tf
from tqdm import tqdm

from .train_utils import Inputs


class RinokerasGraph(ABC):

    _num_graphs = 0

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self._name = self.__class__.__name__.lower()
        if RinokerasGraph._num_graphs > 0:
            self._name += '_{}'.format(RinokerasGraph._num_graphs)
        RinokerasGraph._num_graphs += 1

        self.progress_bar = None
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

        feed_dict = {} # type: Dict[tf.placeholder, Any]
        self._map_to_placeholders(self.inputs, inputs, feed_dict)
        return feed_dict

    def _run_tensor(self, ops: Union[tf.Tensor, Sequence[tf.Tensor]], inputs: Optional[Inputs] = None) -> Any:
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

        results = sess.run(ops, feed_dict=feed_dict)
        return results

    def add_progress_bar(self, data_len: Optional[int] = None, epoch_num: Optional[int] = None):
        desc = None if epoch_num is None else 'Epoch {:>3}'.format(epoch_num)
        progress_bar = tqdm(total=data_len, desc=desc, leave=False,
                            dynamic_ncols=True, smoothing=0.1)
        progress_bar.__enter__()
        self.progress_bar = progress_bar
        return self

    def update_progress_bar(self, postfix=None):
        if self.progress_bar is not None:
            self.progress_bar.update()
            self.progress_bar.set_postfix(postfix)

    def initialize(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.progress_bar is not None:
            self.progress_bar.__exit__()
        self.progress_bar = None
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
    def global_step(self) -> int:
        sess = self._get_session()
        return tf.train.global_step(sess, self._global_step)
