import tensorflow as tf
import numpy.ma as ma
from nltk.translate.bleu_score import corpus_bleu

## Simple tf.pyfunc for computing the bleu score using NLTK's tools
def _masked_bleu_fn_gen(weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False):
    def masked_bleu(reference, hypothesis, reference_mask, hypothesis_mask):
        # First, handle the masking of the functions
        ma_ref = ma.array(reference, mask=reference_mask)
        ma_hyp = ma.array(hypothesis, mask=hypothesis_mask)

        ll_ref_lists = ma_ref.tolist()
        ll_hyp_lists = ma_hyp.tolist()
        ll_ref = [[[str(i) for i in ll if i is not None]] for ll in ll_ref_lists]
        ll_hyp = [[str(i) for i in ll if i is not None] for ll in ll_hyp_lists]
        # Blu score computation
        bleu_score = corpus_bleu(ll_ref, ll_hyp, weights, smoothing_function, auto_reweigh)
        return bleu_score
    
    return masked_bleu

def bleu1(reference, hypothesis, reference_mask=None, hypothesis_mask=None):
    if reference_mask is None:
        reference_mask = tf.ones_like(reference)
    if hypothesis_mask is None:
        hypothesis_mask = tf.ones_like(hypothesis)

    reference_mask = tf.logical_not(tf.cast(reference_mask, tf.bool))
    hypothesis_mask = tf.logical_not(tf.cast(hypothesis_mask, tf.bool))

    return tf.py_func(_masked_bleu_fn_gen((1, 0, 0, 0), None, False), [reference, hypothesis,
                        reference_mask, hypothesis_mask], tf.float32)

def bleu2(reference, hypothesis, reference_mask=None, hypothesis_mask=None):
    if reference_mask is None:
        reference_mask = tf.ones_like(reference)
    if hypothesis_mask is None:
        hypothesis_mask = tf.ones_like(hypothesis)

    reference_mask = tf.logical_not(tf.cast(reference_mask, tf.bool))
    hypothesis_mask = tf.logical_not(tf.cast(hypothesis_mask, tf.bool))

    return tf.py_func(_masked_bleu_fn_gen((0.5, 0.5, 0, 0), None, False), [reference, hypothesis,
                        reference_mask, hypothesis_mask], tf.float32)

def bleu3(reference, hypothesis, reference_mask=None, hypothesis_mask=None):
    if reference_mask is None:
        reference_mask = tf.ones_like(reference)
    if hypothesis_mask is None:
        hypothesis_mask = tf.ones_like(hypothesis)

    reference_mask = tf.logical_not(tf.cast(reference_mask, tf.bool))
    hypothesis_mask = tf.logical_not(tf.cast(hypothesis_mask, tf.bool))

    return tf.py_func(_masked_bleu_fn_gen((0.333, 0.333, 0.333, 0), None, False), [reference, hypothesis,
                        reference_mask, hypothesis_mask], tf.float32)

def bleu4(reference, hypothesis, reference_mask=None, hypothesis_mask=None):
    if reference_mask is None:
        reference_mask = tf.ones_like(reference)
    if hypothesis_mask is None:
        hypothesis_mask = tf.ones_like(hypothesis)

    reference_mask = tf.logical_not(tf.cast(reference_mask, tf.bool))
    hypothesis_mask = tf.logical_not(tf.cast(hypothesis_mask, tf.bool))

    return tf.py_func(_masked_bleu_fn_gen((0.25, 0.25, 0.25, 0.25), None, False), [reference, hypothesis,
                        reference_mask, hypothesis_mask], tf.float32)