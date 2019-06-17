
from typing import Optional
import torch

from rinokeras.core.torch.functional.similarity import scaled_dot_product_similarity
from rinokeras.core.torch.functional.masking import apply_attention_mask


ATTENTION_METHODS_MAP = {
    'scaled_dot': scaled_dot_product_similarity
}
ATTENTION_FUNCTION_MAP = {
    'softmax': torch.nn.functional.softmax,
}

def attention_map(queries: torch.Tensor,
                  keys: torch.Tensor,
                  values: torch.Tensor,
                  mask: torch.Tensor = None,
                  dropout: Optional[float] = None,
                  return_attention_weights: bool = True,
                  similarity_metric: str = 'scaled_dot',
                  attention_function: str = 'softmax') -> torch.Tensor:

    similarity = ATTENTION_METHODS_MAP[similarity_metric](queries, keys)

    if attention_function == 'softmax':
        masked_similarity = apply_attention_mask(similarity, mask=mask)
        weights = torch.nn.functional.softmax(masked_similarity - torch.max(masked_similarity, dim=-1, keepdim=True)[0], dim=-1)
    else:
        masked_similarity = apply_attention_mask(similarity, mask=mask, hadamard=True)
        weights = ATTENTION_FUNCTION_MAP[attention_function](masked_similarity, dim=-1)

    if dropout:
        weights = torch.nn.functional.dropout(weights, dropout)
    outputs = torch.matmul(weights, values)
    if return_attention_weights:
        return outputs, weights
    return outputs

def split_heads(input_tensor: torch.Tensor, n_heads:int) -> torch.Tensor:
    # Splits the last dimension into a heads dimension
    if input_tensor.shape[-1] % n_heads != 0:
        raise AssertionError('Tensor shape at dimension -1 ({}) must be divisible by n_heads ({})'.format(input_tensor.shape[-1], n_heads))
    if len(input_tensor.shape) != 3:
        raise AssertionError('Input to split_heads must be rank 3')
    
    output = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], n_heads, input_tensor.shape[2]//n_heads)
    return output.permute(0,2,1,3)

def combine_heads(input_tensor: torch.Tensor) -> torch.Tensor:
    if len(input_tensor.shape) != 4:
        raise AssertionError('Input to combine_heads must be rank 4')
    output = input_tensor.permute(0,2,1,3)
    return output.reshape(input_tensor.shape[0], input_tensor.shape[2], -1)

def multi_head_attention_map(queries: torch.Tensor,
                             keys: torch.Tensor,
                             values: torch.Tensor,
                             n_heads: int,
                             mask: torch.Tensor = None,
                             dropout: Optional[float] = None,
                             return_attention_weights: bool = True,
                             similarity_metric: str = 'scaled_dot',
                             attention_function: str = 'softmax') -> torch.Tensor:

    queries_split = split_heads(queries, n_heads)
    keys_split = split_heads(keys, n_heads)
    values_split = split_heads(values, n_heads)

    attention_outputs, attention_weights = attention_map(queries=queries_split,
                                                        keys=keys_split,
                                                        values=values_split,
                                                        mask=mask,
                                                        dropout=dropout,
                                                        return_attention_weights=True,
                                                        similarity_metric=similarity_metric,
                                                        attention_function=attention_function)

    outputs = combine_heads(attention_outputs)

    if return_attention_weights:
        return outputs, attention_weights
    return outputs
