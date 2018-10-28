from abc import ABC, abstractmethod
from typing import Sequence, Union, Any, Optional, Callable, Tuple, List

import tensorflow as tf
from tqdm import tqdm


class RinokerasGraph(ABC):

    _num_graphs: int = 0

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        self._name = self.__class__.__name__.lower()
        if RinokerasGraph._num_graphs > 0:
            self._name += '_{}'.format(RinokerasGraph._num_graphs)
        RinokerasGraph._num_graphs += 1

        self.progress_bar = None

    def _unpack_losses(self, losses: Union[tf.Tensor, Sequence[tf.Tensor]]):
        """Optionally unpacks a sequence of losses

        Args:
            losses (Union[tf.Tensor, Sequence[tf.Tensor]]): Loss tensor or sequence of loss tensors with
                first tensor being total loss

        Returns:
            tf.Tensor, Union[tf.Tensor, Sequence[tf.Tensor]]: Total loss, and sequence of loss tensors
        """
        if isinstance(losses, tuple) or isinstance(losses, list):
            total_loss = losses[0]
        else:
            total_loss = losses
            losses = (losses,)

        return total_loss, losses

    def _get_gradient_clip_function(self, clip_type: str, clip_bounds: Union[float, Tuple[float, ...]]) -> \
            Callable[[Sequence], List]:

        def clip_func(grads):
            clipped_grads = []
            for g, v in grads:
                if g is None:
                    # Choosing not to add gradients to list if they're None. Both adding/not adding are valid choices.
                    # clipped_grads.append((None, v))
                    continue
                if not v.trainable:
                    continue
                if clip_type in ['none', 'None']:
                    pass
                elif clip_type == 'value':
                    g = tf.clip_by_value(g, clip_bounds[0], clip_bounds[1])
                elif clip_type == 'norm':
                    g = tf.clip_by_norm(g, clip_bounds)
                elif clip_type == 'global_norm':
                    g = tf.clip_by_global_norm(g, clip_bounds)
                elif clip_type == 'average_norm':
                    g = tf.clip_by_average_norm(g, clip_bounds)
                else:
                    raise ValueError("Unrecognized gradient clipping method: {}.".format(clip_type))
                clipped_grads.append((g, v))
            return clipped_grads
        return clip_func

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.progress_bar.__exit__()
        self.progress_bar = None
        return exc_type is None or exc_type == tf.errors.OutOfRangeError

    @abstractmethod
    def run(self, ops: Union[str, tf.Tensor, Sequence[tf.Tensor]], *args, **kwargs) -> Any:
        raise NotImplementedError("run op not implemented.")
