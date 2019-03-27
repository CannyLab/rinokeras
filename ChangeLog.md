# Rinokeras 0.2 -- Changelog

## Major Changes (Completed)
- Moved tensorflow 1.x rinokeras to rinokeras.v1x
- Added support for tensorflow 1.13

## Major Changes (In Progress):
- Added additional testing (regression + unit testing) for 1.x code
- API changed to conform to Keras standards
- Add back-support for tensorflow < 1.13 (<= 1.12)

## Major Changes (Planned)
- Changed default to tensorflow 2.0
- Added support for tensorflow 2.0
- Added TPU support for tensorflow 2.0 libraries
- Add support for Keras compile/fit
- Restructure utils folder

## Minor/All Changes

### Graphs (1.x)
- Changed batch_reduce calls to batch_all_reduce calls
- Changed call_for_each_tower calls to call_for_each_replica calls
- Added testing for graph generation and execution

### Attention Layers (1.x)
- Luong attention changed so local=True variant takes a tuple of three inputs (target_hidden, source_hidden, position)
- AttentionQKV renamed to AttentionQKVProjection, altered so that it takes (query, key, value) antecedents instead of (query, memory) antecedents
- AttentionQKVProjection now takes an optional project_value argument which turns off the value projection
- Trilinear similarity changed so the pass/return is (context, query) - formerly the tuple passed was (query, context) and returned was a tensor with [bs x context x query]
- Altered Trilinear similarity to support multiple heads in the case of using it in an attention layer directly (instead of in CQ attention)
- Changed ScaledDotProductSimilarity to match API of TrilinearSimilarity by taking a tuple (queries, keys) instead of two independent arguments
- Keyword of ApplyAttentionMask changed from "similarity" to "inputs" to reflect Keras API requirements
- Fixed bug in ApplyAttentionMask for non-softmax losses
- AttentionMap changed to comply with Keras API, now takes "inputs" tuple of (queries, keys, values) instead of queries, keys, values as keywords
- Fixed bug in AttentionMap when using differing numbers of keys and queries
- Added return_attention_weights argument to AttentionMap layer
- Added attention_function argument to MultiHeadAttentionMap layer
- Added trilinear similarity to MultiHeadAttention layer
- Added attention_function argument to MultiHeadAttention layer
- Changed MultiHeadAttention layer to take separate query, key and value antecedents
- Changed MultiHeadAttention layer to take/pass project_value argument to AttentionQKVProjection
- Changed ContextQueryAttention layer to take tuple of (context, query) as "inputs"
- Reversed order of "context" and "query" in call to ContextQueryAttention to maintain consistency
- Changed ContextQueryAttention "attention_method" argument to "similarity_metric" to maintain consistency


