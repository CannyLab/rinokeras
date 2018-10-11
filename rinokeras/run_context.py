import sys
from abc import ABC
from typing import Optional, Sequence
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tqdm import tqdm


class epoch(ABC):

    def __init__(self,
                 trainer,
                 dataset: tf.data.Dataset,
                 epoch: int,
                 data_len: Optional[int] = None,
                 loss_names: Optional[Sequence[str]] = None,
                 summary_writer: Optional[tf.summary.FileWriter] = None,
                 sess: Optional[tf.Session] = None,
                 summary_iter: Optional[int] = 1000) -> None:
        self.trainer = trainer
        self.dataset = dataset
        self.iterator = dataset.make_one_shot_iterator()
        self.epoch = epoch
        self.data_len = data_len if data_len is not None else 0
        self.loss_names = loss_names
        self.summary_writer = summary_writer
        self.summary_iter = summary_iter
        self.sess = sess if sess is not None else tf.get_default_session()
        if self.sess is None:
            raise RuntimeError("No tensorflow session detected.")
        self.handle = self.sess.run(self.iterator.string_handle())

        self.n_minibatches = 0
        self.losses = None
        self.start = timer()

    def __enter__(self):
        self.progress_bar = tqdm(total=self.data_len, desc='Epoch {:>3}'.format(self.epoch), leave=False,
                                 dynamic_ncols=True, smoothing=0.1)
        self.progress_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.progress_bar.__exit__()
        return exc_type is None or exc_type == tf.errors.OutOfRangeError

    def process_iteration(self, losses, summary):
        if self.losses is None:
            self.losses = np.array(losses)
        else:
            self.losses += np.array(losses)

        self.n_minibatches += 1
        if self.loss_names is not None:
            postfix = {ln: loss / self.n_minibatches for ln, loss in zip(self.loss_names, self.losses)}
        else:
            postfix = {'Loss{}'.format(i): loss / self.n_minibatches for i, loss in enumerate(losses)}
        self.progress_bar.update()
        self.progress_bar.set_postfix(postfix)

        if self.n_minibatches % self.summary_iter == 0 and self.summary_writer is not None:
            self.summary_writer.add_summary(summary, self.epoch * self.data_len + self.n_minibatches)


class train_epoch(epoch):

    def run_iteration(self):
        losses, summary = self.trainer.train(self.handle)
        self.process_iteration(losses, summary)
        return self.losses / self.n_minibatches, self.n_minibatches


class test_epoch(epoch):

    def run_iteration(self):
        losses, summary = self.trainer.loss(self.handle)
        self.process_iteration(losses, summary)
        return self.losses / self.n_minibatches, self.n_minibatches


# def run_graph_epoch(policy, dataset, is_training, epoch, data_len, summary_writer):
#     avgloss = None
#     n_minibatches = 0
#     data_iterator = dataset.make_one_shot_iterator()
#     data_handle = sess.run(data_iterator.string_handle())
#     start = timer()
#     with tqdm(total=data_len, desc='Epoch {:>3}'.format(epoch),
#               leave=False, dynamic_ncols=True, smoothing=0.1) as progress_bar:
#         while True:
#             try:
#                 if is_training:
#                     currloss, summary = trainer.train(data_handle)
#                 else:
#                     currloss, summary = trainer.loss(data_handle)

#                 if summary_writer is not None:
#                     summary_writer.add_summary(summary, epoch * data_len + n_minibatches)

#                 if avgloss is None:
#                     avgloss = 0.0 if len(currloss) == 1 else np.zeros((len(currloss) - 1,))

#                 if len(currloss) > 1:
#                     avgloss += np.array(currloss[1:])
#                 else:
#                     avgloss += currloss[0]

#                 n_minibatches += 1
#                 progress_bar.update()

#                 disploss = avgloss / n_minibatches
#                 if np.isscalar(disploss):
#                     disploss = (disploss,)

#                 postfix = {name: loss for name, loss in zip(names, disploss)}

#                 progress_bar.set_postfix(postfix)
#             except tf.errors.OutOfRangeError:
#                 break
#     loss_final = avgloss / n_minibatches
#     if np.isscalar(loss_final):
#         loss_final = (loss_final,)
#     time = timer() - start
#     return loss_final, time, n_minibatches
