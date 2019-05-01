
import torch

def scaled_dot_product_similarity(querys: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    similarity = torch.matmul(querys, keys.transpose(-1, -2)) / torch.sqrt(keys.shape[-1])
    return similarity
