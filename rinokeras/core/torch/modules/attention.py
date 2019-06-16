"""
Attention layers
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rinokeras.core.torch.utils.tensor import get_parameter

from rinokeras.core.torch.functional.masking import apply_attention_mask, convert_sequence_mask_to_attention_mask
from rinokeras.core.torch.functional.attention import multi_head_attention_map



class LuongAttention(nn.Module):
    def __init__(self, source_dim: int, target_dim: int, output_units: int, stddev: float = 1.0) -> None:
        super(LuongAttention, self).__init__()
        self.stddev = stddev
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.output_units = output_units

        self.attention_weights = get_parameter([self.source_dim + self.target_dim, self.output_units])

    def forward(self,source_hidden_sequence: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        # Source Hidden Sequence -> Tensor [None, None, encoder_cell_size]
        # Target Hidden -> Tensor [None, decoder_cell_size]
        attention_score = torch.matmul(source_hidden_sequence, target_hidden.unsqueeze(-1))
        weights = F.softmax(attention_score, dim=1)
        weighted = (source_hidden_sequence * weights).sum(dim=1)
        return F.tanh(torch.matmul(torch.cat([target_hidden, weighted], dim=1), self.attention_weights))

class LocalLuongAttention(nn.Module):
    def __init__(self, source_dim: int, target_dim: int, output_units: int, stddev: float = 1.0) -> None:
        super(LocalLuongAttention, self).__init__()
        self.stddev = stddev
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.output_units = output_units

        self.attention_weights = get_parameter([self.source_dim + self.target_dim, self.output_units])

    def forward(self,source_hidden_sequence: torch.Tensor, target_hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Source Hidden Sequence -> Tensor [None, None, encoder_cell_size]
        # Target Hidden -> Tensor [None, decoder_cell_size]
        attention_score = torch.matmul(source_hidden_sequence, target_hidden.unsqueeze(-1))
        weights = F.softmax(attention_score, dim=1)

        # Handle the locality of the attention
        relative_position = torch.range(source_hidden_sequence.shape[1]) - positions.reshape(-1, 1)
        position_weighting = torch.exp(-1 * torch.pow(relative_position, 2) / (2 * torch.pow(self.stddev, 2)))
        weights = weights * position_weighting.unsqueeze(-1)

        weighted = (source_hidden_sequence * weights).sum(dim=1)
        return F.tanh(torch.matmul(torch.cat([target_hidden, weighted], dim=1), self.attention_weights))


class AttentionQKVProjection(nn.Module):
    def __init__(self, query_input_dim: int,
                 key_input_dim: int,
                 value_input_dim: int,
                 key_dim: int,
                 value_dim: int,
                 project_value: bool = True):
        super(AttentionQKVProjection, self).__init__()

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.project_value = project_value
        
        self.query_input_dim = query_input_dim
        self.key_input_dim = key_input_dim
        self.value_input_dim = value_input_dim

        # Setup the projection layers
        self.query_layer = nn.Linear(self.query_input_dim, self.key_dim)
        self.query_norm = nn.LayerNorm(self.key_dim)

        self.key_layer = nn.Linear(self.key_input_dim, self.key_dim)
        self.key_norm = nn.LayerNorm(self.key_dim)


        if project_value:
            self.value_layer = nn.Linear(self.value_input_dim, self.value_dim)
            self.value_norm = nn.LayerNorm(self.value_dim)

    def forward(self, query_antecedent, key_antecedent, value_antecedent) -> torch.Tensor:
        q = self.query_norm(self.query_layer(query_antecedent))
        k = self.key_norm(self.key_layer(key_antecedent))
        v = self.value_norm(self.value_layer(value_antecedent)) if self.project_value else value_antecedent
        return q,k,v

class TrilinearSimilarity(nn.Module):

    def __init__(self,
                 query_input_dim: int,
                 context_input_dim: int,
                 dropout: Optional[float] = None) -> None:
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.query_weights = get_parameter([query_input_dim, 1])
        self.context_weights = get_parameter([context_input_dim, 1])
        self.dot_weights = get_parameter([context_input_dim, query_input_dim])

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:

        # TODO: Why are these here? Seems like an odd modeling decision

        # Context is [batch_size x context_length x context_channels]
        # Query is [batch_size x (n_heads) x query_length x query_channels]
        context = self.dropout(context)
        query = self.dropout(query)

        # Context_Weighted is [batch_size x context_length x 1]
        context_weighted = torch.matmul(context, self.context_weights)


        # Query weighted is [batch_size x (n-heads) x 1 x query_length]
        if len(query.shape) == 3:
            query_weighted = torch.matmul(query, self.query_weights).permute(0, 2, 1)
        if len(query.shape) == 4:
            # We need to account for the heads in the multi-head attention case
            query_weighted = torch.matmul(query, self.query_weights).permute(0, 1, 3, 2)

        # Weighted Context Query is [batch_size x context_length x query_length]
        weighted_context_query = torch.matmul(torch.matmul(context, self.dot_weights), query.transpose(-1, -2))
        output = weighted_context_query + query_weighted  + context_weighted
        return output
        
class MultiHeadAttention(nn.Module):

    def __init__(self,
                 query_input_dim: int,
                 key_input_dim: int,
                 value_input_dim: int,
                 n_heads: int,
                 dropout: Optional[float] = None,
                 key_dim: Optional[int] = None,
                 value_dim: Optional[int] = None,
                 attention_function: str = 'softmax',
                 project_value: bool = True,
                 similarity_metric: str = 'scaled_dot') -> None:
        super(MultiHeadAttention, self).__init__()

        self.query_input_dim = query_input_dim
        self.key_input_dim = key_input_dim
        self.value_input_dim = value_input_dim
        
        self.key_dim = key_dim if key_dim else key_input_dim
        self.value_dim = value_dim if value_dim else value_input_dim

        self.project_value = project_value
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.attention_function = attention_function
        self.similarity_metric = similarity_metric

        # Setup the projection and output layers
        self.qkv_projection = AttentionQKVProjection(query_input_dim=self.query_input_dim,
                                                     key_input_dim=self.key_input_dim,
                                                     value_input_dim=self.value_input_dim,
                                                     key_dim=self.key_dim,
                                                     value_dim=self.value_dim,
                                                     project_value=self.project_value)
        self.output_layer = torch.nn.Linear(self.value_dim, self.value_dim)

        # Setup extra layers
        self.dropout = torch.nn.Dropout(dropout) if dropout else None

    def forward(self,
                query_antecedent: torch.Tensor,
                key_antecedent: torch.Tensor,
                value_antecedent: torch.Tensor,
                mask: torch.Tensor = None,
                return_attention_weights: bool = False) -> torch.Tensor:
        q, k, v = self.qkv_projection( query_antecedent, key_antecedent, value_antecedent)
        attention_outputs, attention_weights = multi_head_attention_map(q,
                                                                        k,
                                                                        v,
                                                                        self.n_heads,
                                                                        mask=mask,
                                                                        return_attention_weights=True,
                                                                        attention_function=self.attention_function,
                                                                        similarity_metric=self.similarity_metric,
                                                                        dropout=self.dropout_rate)

        output = self.output_layer(attention_outputs)
        if self.dropout:
            output = self.dropout(attention_outputs)
        if return_attention_weights:
            return output, attention_weights
        return output

class SelfAttention(nn.Module):
    
    def __init__(self,
                input_dim: int,
                n_heads: int,
                dropout: Optional[float] = None,
                key_dim: Optional[int] = None,
                value_dim: Optional[int] = None,
                attention_function: str = 'softmax',
                project_value: bool = True,
                similarity_metric: str = 'scaled_dot') -> None:

        super(SelfAttention, self).__init__()

        self.multi_attention = MultiHeadAttention(query_input_dim=input_dim,
                                                    key_input_dim=input_dim,
                                                    value_input_dim=input_dim,
                                                    n_heads=n_heads,
                                                    dropout=dropout,
                                                    key_dim=key_dim,
                                                    value_dim=value_dim,
                                                    attention_function=attention_function,
                                                    project_value=project_value,
                                                    similarity_metric=similarity_metric)
        
    def forward(self,
                input_sequence: torch.Tensor,
                mask: torch.Tensor = None,
                return_attention_weights: bool = True) -> torch.Tensor:
        return self.multi_attention(input_sequence,
                                    input_sequence,
                                    input_sequence,
                                    mask=mask,
                                    return_attention_weights=return_attention_weights)


class ContextQueryAttention(nn.Module):

    def __init__(self,
                query_input_dim: int,
                context_input_dim: int,
                dropout: Optional[float] = None) -> None:
        super(ContextQueryAttention, self).__init__()
        self.dropout = torch.nn.Dropout(dropout) if dropout else None
        self.similarity = TrilinearSimilarity(query_input_dim, context_input_dim, dropout=dropout)

    def forward(self, context: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        similarity = self.similarity(context, query)
        masked_similarity = apply_attention_mask(similarity, mask)

        c2q_sim = F.softmax(masked_similarity, dim=-1)
        q2c_sim = F.softmax(masked_similarity, dim=-2)

        c2q = torch.matmul(c2q_sim, query)
        q2c = torch.matmul(torch.matmul(c2q_sim, q2c_sim.transpose(-1,-2)), context)

        outputs = torch.cat([context, c2q, context * c2q, context * q2c], dim=-1)
        if self.dropout:
            outputs = self.dropout(outputs)
        return outputs


class StridedCachedLWSelfAttention(nn.Module):
    """
    Limited window self-attention block which can be used for the implementation
    of the transformer-XL architecture.

    Note that this has to be called sequentially. It's going to have issues if 
    you don't do this -> There's a bunch of clever data management that you have
    to do, and you have to make sure that you re-initialize the cache when you
    transition to a new document. Pre-warming the cache with some other values
    is potential future research.
    """

    def __init__(self, degree: int, stride: int, model_dim: int, n_heads: int = 4) -> None:

        super(StridedCachedLWSelfAttention, self).__init__()

        # Initialize the update elements
        self.degree = degree
        self.stride = stride
        self.model_dim = model_dim
        self.n_heads = n_heads
        
        # Attention
        self.attention_layer = SelfAttention(model_dim, self.n_heads)

    def get_causal_mask(self, sequence):
        ranges = torch.arange(0, sequence.shape[1]) # Get the ranges 1...seqlen
        output = ranges.view(1,-1).expand(sequence.shape[1], -1)
        output = (output <= ranges.view(-1, 1).expand(-1, sequence.shape[1])).expand(sequence.shape[0], -1, -1).cuda()
        return output.squeeze(-1)

    def forward(self, inputs: torch.Tensor, sequence_mask: torch.Tensor = None, cache: torch.Tensor = None) -> torch.Tensor:

        if inputs.shape[1] != self.degree*self.stride:
            raise AssertionError('Inputs must be of shape [{},{},{}]'.format(self.last_batch_size, self.degree*self.stride, self.model_dim))

        # Setup the cache
        if cache is None:
            cache = torch.zeros(self.stride, inputs.shape[0], self.degree, self.model_dim, requires_grad=False)
            cache = cache.to(inputs.device) # Move the cache to the correct device

        if self.stride != 1:
            return_tensor = torch.zeros_like(inputs)
            for s in range(self.stride):
                s_inputs = inputs[:, s::self.stride, :]
                

                # Cat the cache to the current inputs -> This should give
                self_attention_inputs = torch.cat([cache[s].detach(), s_inputs], dim=1)
                # Get the masking set up properly
                self_attention_mask = self.get_causal_mask(self_attention_inputs)

                if sequence_mask is not None:
                    s_mask = sequence_mask[:,s::self.stride]
                    # Expand the sequence mask for the cache
                    s_mask = torch.nn.functional.pad(s_mask, (cache.shape[2], 0), value=1)
                    sa_mask = convert_sequence_mask_to_attention_mask(self_attention_inputs, s_mask)

                    self_attention_mask = sa_mask.float() * self_attention_mask.float()

                attn_out = self.attention_layer(self_attention_inputs, self_attention_mask, return_attention_weights=False)
                attn_out_slices = attn_out[:, self.degree:, :]
                cache[s] = attn_out_slices
                return_tensor[:, s::self.stride, :] = attn_out_slices
            
            sliced_out = return_tensor
        else:
            # Cat the cache to the current inputs -> This should give
            self_attention_inputs = torch.cat([cache[0].detach(), inputs], dim=1)

            # Get the masking set up properly
            self_attention_mask = self.get_causal_mask(self_attention_inputs)
            if sequence_mask is not None:
                sequence_mask = torch.nn.functional.pad(sequence_mask, (cache.shape[2], 0),value=1)
                sa_mask = convert_sequence_mask_to_attention_mask(self_attention_inputs, sequence_mask)
                self_attention_mask = sa_mask.float() * self_attention_mask.float()

            attention_outputs = self.attention_layer(self_attention_inputs, self_attention_mask, return_attention_weights=False)

            # Update the cache
            sliced_out = attention_outputs[:, self.degree:, :]
            cache[0] = sliced_out

        return sliced_out, cache
