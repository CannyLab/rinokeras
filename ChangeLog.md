# Rinokeras 0.2 -- Changelog

## Major Changes (Completed)
- Moved tensorflow 1.x rinokeras to rinokeras.core.v1x
- Updated folder structure for cleaner API changes
- Added support for tensorflow 1.13
- Confirmed support for tensorflow 1.12
- Confirmed support in Python 3.4,3.5,3.6,3.7

## Major Changes (In Progress):
- Added additional testing (regression + unit testing) for 1.x code
- API changed to conform to Keras standards
- Added support for tensorflow 2.0
- Restructure utils folder

## Major Changes (Planned)
- Changed default to tensorflow 2.0
- Added TPU support for tensorflow 2.0 libraries
- Add support for Keras compile/fit


## Minor/All Changes

### Continuous Integration
- Added support for code coverage
- Increased supported testing from Python 3.6 -> Python 3.4, 3.5, 3.6 and 3.7
- Bumped testing VM to Ubuntu 16.04 (From 14.04)

### Library
- Removed outdated examples
- Simplified API
    - rinokeras.common.layers moved to rinokeras.layers
    - rinokeras.common.distributions moved to rinokeras.layers.distributions
    - rinokeras.common.gcn moved to rinokeras.layers.gcn
    - rinokeras.common.losses moved to rinokeras.losses
    - rinokeras.common.attention moved to rinokeras.layers.attention
    - rinokeras.common.optimizers moved to rinokeras.train.optimizers
    - rinokeras.common.rnn moved to rinokeras.models.rnn
- Removed non-functioning code
    - Updated GLOW model to be functional
    - Updated DenseTranspose to be functional
    - Removed Adamax
- Deprecated EagerGraph v1x API in favor of Rinokeras v2x
- Moved GroupedConvolution and ResidualBlock from rinokeras.models.resnet to rinokeras.layers.conv and rinokeras.layers.residual respectively

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
- Added get_config/from_config to SelfAttention, MultiHeadAttention

### Activation Layers (1.x)
- No Changes

### Autoregressive Layers (1.x)
- Fixed breaking bug in CouplingLayer

### Conv Layers (1.x)
- Renamed NormedConv to NormedConvStack to maintain naming consistency
- Added GroupedConv from rinokeras.models.resnet

### Distribution Layers (1.x)
- Fixed bug in entropy of CategoricalPd

### Dropout Layers (1.x)
- Fixed bug in LayerDropout where layer was never dropped (even during training)

### (NEW) Embedding Layers (1.x)
- Moved VQAImageEmbedding to ImageEmbedding layer

### GCN (1.x)
- Fixed shape bugs in GCN
- Changed GCN so that inputs are [BS x N x units] instead of [BS x units x N]

### Inversion Layers (1.x)
- Fixed name bug in DenseTranspose

### Losses (1.x)
- No Changes

### Masking Layers (1.x)
- Renamed MaskInput to BERTRandomReplaceMask

### Normalization Layers (1.x)
- No Changes

### Position Embedding Layers (1.x)
- Fixed bug with reproject_embedding in 2D and 3D cases
- Fixed bug with concatenation in 3D case

### Residual Layers (1.x)
- Changed Highway layer to take a layer as input (expanded functionality)
- Added ResidualBlock from rinokeras.models.resnet

### Stacks (1.x)
- Added LayerDropoutStack which adds layer-dropout between each layer of the stack

### Transformer (1.x)
- Moved TransformerSelfAttention and TransformerMultiAttention into transformer_attention.py
- Moved TransformerFeedForward into transformer_ff.py
- Moved TransformerEncoderBlock and TransformerEncoder into transformer_encoder.py
- Moved TransformerDecoderBlock and TransformerDecoder into transformer_decoder.py
- Moved TransformerInputEmbedding into transformer_embedding.py
- Simplified TransformerSelfAttentionOnlyDecoder (TASODecoder) into transformer_decoder (just pass None for the source)
- Changed TransformerMultiAttention to conform to keras inputs, now takes input tuple inputs=(source, target)
- Updated parameter ordering of TransformerFeedForward layer
- Updated parameter ordering of TransformerEncoderBlock layer
- Changed TransformerEncoder to take tuple mask=(encoder_mask, conv_mask)
- Changed TransformerDecoderBlock API to conform to keras inputs, now takes inputs tuple=(encoder_inputs, decoder_inputs) and mask tuple=(self_attention_mask, cross_attention_mask)
- Updated parameter ordering of TransformerDecoderBlock layer
- Updated parameter odering of TransformerDeocder layer
- Changed TransformerDecoder API to conform to keras inputs, now takes inputs tuple=(encoder_inputs, decoder_inputs) and mask tuple=(self_attention_mask, cross_attention_mask)
- Changed fast_beam_decode to take n_beams instead of beam_size
- Fixed bugs in fast_decode for non-discrete outputs
- Added get_config/from_config to TransformerSelfAttention, TransformerMultiAttention, TransformerFeedForward, TransformerEncoderBlock, TransformerEncoder, TransformerDecoderBlock, TransformerDecoder and Transformer
- Refactored some utilities from transformer_decoder into transformer_utils
- Updated transformer to conform to Keras API

### QANet (1.x)
- Refactored QANet into multiple files
- Fixed bug in QANet self attention
- Reordered arguments in QANet/QANetEncoder
- Added get_config/from_config to all QANet layers
- Added support for random (learned) embedding matrices

### ResNet (1.x)
- Moved GroupedConv to Conv layers

### GLOW (1.x)
- Added warnings to GLOW code about functionality

# Known Errors/Bugs/Questions

### Distribution Layers (1.x)
- Odd shape for entropy of DiagGaussPd () vs (BS,)

### Transformer (1)
- Using a custom output layer may cause problems when reconstructing from config (untested)

### QANet (1.x)
- QANet layers don't handle extra kwargs in get/from config



