# Changelog

## Rinokeras 0.2

### Major Changes:
- Changed default to tensorflow 2.0
- Moved tensorflow 1.x rinokeras to rinokeras.v1x
- Added support for tensorflow 1.13
- Added support for tensorflow 2.0
- Added TPU support for tensorflow 2.0 libraries
- API made more consistent

### Minor/All Changes

##### Attention Layers
- Luong attention changed so local=True variant takes a tuple of three inputs (target_hidden, source_hidden, position)
- AttentionQKV renamed to AttentionQKVProjection, altered so that it takes (query, key, value) antecedents instead of (query, memory) antecedents
- Trilinear similarity changed so the pass/return is (context, query) - formerly the tuple passed was (query, context) and returned was a tensor with [bs x context x query]
