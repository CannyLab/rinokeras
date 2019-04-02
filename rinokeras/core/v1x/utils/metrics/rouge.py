import tensorflow as tf
import numpy.ma as ma
from .pyrouge import Rouge

## Simple tf.pyfunc for computing the rouge score using PyRouge
def masked_rouge(reference, hypothesis, reference_mask, hypothesis_mask):
    # First, handle the masking of the functions
    ma_ref = ma.array(reference, mask=reference_mask)
    ma_hyp = ma.array(hypothesis, mask=hypothesis_mask)

    ll_ref_lists = ma_ref.tolist()
    ll_hyp_lists = ma_hyp.tolist()
    ll_ref = [[str(i) for i in ll if i is not None] for ll in ll_ref_lists]
    ll_hyp = [[str(i) for i in ll if i is not None] for ll in ll_hyp_lists]

    rouge_computer = Rouge()
    precision, recall, F_score = rouge_computer.rouge_l(ll_hyp, ll_ref)
    return F_score

def rouge_l(reference, hypothesis, reference_mask=None, hypothesis_mask=None):
    if reference_mask is None:
        reference_mask = tf.ones_like(reference)
    if hypothesis_mask is None:
        hypothesis_mask = tf.ones_like(hypothesis)

    reference_mask = tf.logical_not(tf.cast(reference_mask, tf.bool))
    hypothesis_mask = tf.logical_not(tf.cast(hypothesis_mask, tf.bool))

    return tf.py_func(masked_rouge, [reference, hypothesis,
                        reference_mask, hypothesis_mask], tf.float32)