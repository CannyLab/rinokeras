
import torch
from rinokeras.core.torch.functional.masking import convert_sequence_length_to_sequence_mask

def sequence_loss(inputs, sequence_lengths, outputs, vocab_size):
    # Get the seqeunce mask
    mask_weights = convert_sequence_length_to_sequence_mask(inputs, sequence_lengths)[:, 1:]
    # Shuft the labels left
    targets = inputs[:,1:].reshape(-1)
    logits = outputs[:,:-1].reshape(-1, vocab_size)
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    loss *= mask_weights.reshape(-1)
    loss = loss.sum() / sequence_lengths.sum()
    return loss
