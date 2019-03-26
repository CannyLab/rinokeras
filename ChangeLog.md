# Rinokeras 0.2 -- Changelog

## Major Changes (Completed):
- Moved tensorflow 1.x rinokeras to rinokeras.v1x
- Added support for tensorflow 1.13
- Added additional testing (regression + unit testing) for 1.x code
- API changed to conform to Keras standards

## Major Changes (In Progress)
- Changed default to tensorflow 2.0
- Added support for tensorflow 2.0
- Added TPU support for tensorflow 2.0 libraries

## Minor/All Changes

### Attention Layers (1.x)
- Luong attention changed so local=True variant takes a tuple of three inputs (target_hidden, source_hidden, position)
- AttentionQKV renamed to AttentionQKVProjection, altered so that it takes (query, key, value) antecedents instead of (query, memory) antecedents
- Trilinear similarity changed so the pass/return is (context, query) - formerly the tuple passed was (query, context) and returned was a tensor with [bs x context x query]
- Changed ScaledDotProductSimilarity to match API of TrilinearSimilarity by taking a tuple (queries, keys) instead of two independent arguments
- Keyword of ApplyAttentionMask changed from "similarity" to "inputs" to reflect Keras API requirements
- Fixed bug in ApplyAttentionMask for non-softmax losses
- AttentionMap changed to comply with Keras API, now takes "inputs" tuple of (queries, keys, values) instead of queries, keys, values as keywords
- Fixed bug in AttentionMap when using differing numbers of keys and queries

