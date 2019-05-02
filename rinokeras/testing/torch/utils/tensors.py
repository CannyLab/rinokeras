
import torch

def random_mask_tensor(*shapes):
    bd = torch.distributions.Bernoulli(0.5)
    return bd.sample(shapes)
