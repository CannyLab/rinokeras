
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from typing import Optional

from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Lambda, Dropout, BatchNormalization

from rinokeras.core.v1x.common.layers import WeightNormDense as Dense
from rinokeras.core.v1x.common.layers import DenseStack, PositionEmbedding


class TransformerInputEmbedding(Model):

    def __init__(self,
                 embed_size: int,
                 discrete: bool,
                 n_symbols: Optional[int] = None,
                 dropout: Optional[float] = None,
                 batch_norm: bool = False,
                 n_embed_layers: int = 1,
                 embedding_initializer=None,
                 freeze_embeddings=False,
                 use_position_encoding=True,
                 concat_position_encoding=False,
                 reproject_position_encoding=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.n_symbols = n_symbols
        self.n_embed_layers = n_embed_layers
        self.embedding_initializer = embedding_initializer
        self.embedding_dense = Lambda(lambda x: x)
        self.using_dense_embedding = False

        self.use_position_encoding = use_position_encoding
        self.concat_position_encoding = concat_position_encoding
        self.reproject_position_encoding = reproject_position_encoding

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        if discrete:
            assert n_symbols is not None, 'n_symbols not passed in but model set to discrete'
            assert n_embed_layers == 1, 'discrete models can only have one embedding layer'

            if embedding_initializer is not None:
                assert embedding_initializer.shape[0] == n_symbols, \
                    'n_symbols and initializer shape mismatch'

                if embedding_initializer.shape[1] != embed_size:
                    # We have to correct if the input embedding isn't quite right
                    self.embedding = Embedding(n_symbols, embedding_initializer.shape[1],
                                               weights=[embedding_initializer],
                                               trainable=not freeze_embeddings)
                    self.embedding_dense = Dense(embed_size)
                    self.using_dense_embedding = True
                else:
                    self.embedding = Embedding(n_symbols, embed_size,
                                               weights=[embedding_initializer])
            else:
                self.embedding = Embedding(n_symbols, embed_size)
        else:
            assert n_symbols is None, 'n_symbols passed in but model set to continuous'
            assert embedding_initializer is None, 'embedding_initializer passed in but model set to continouous'
            self.embedding = DenseStack([embed_size] * n_embed_layers, output_activation='relu',
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer)

        self.discrete = discrete
        self.freeze_embeddings = freeze_embeddings
        self.position_encoding = PositionEmbedding(
            concat=self.concat_position_encoding, reproject_embedding=reproject_position_encoding)
        self.dropout_rate = dropout
        self.dropout = Dropout(0 if dropout is None else dropout)
        self.use_batch_norm = batch_norm
        self.batch_norm = None if batch_norm is False else BatchNormalization()

    def call(self, inputs, start=1):
        # Compute the actual embedding of the inputs by using the embedding layer
        # TODO: Make sure that for non-discrete embeddings, this is handled correctly
        # and allow the shape to be correctly sorted. This should have a tensor
        # as output with shape [batch_size x sequence_len x d_model]
        embedding = self.embedding(inputs)

        if self.freeze_embeddings:
            embedding = K.stop_gradient(embedding)
        embedding = self.embedding_dense(embedding)
        embedding = self.dropout(embedding)

        if self.batch_norm:
            embedding = self.batch_norm(embedding)
        if self.use_position_encoding:
            embedding = self.position_encoding(embedding, start=start)

        return embedding

    def get_config(self):
        ei = self.embedding_initializer.tolist() if self.embedding_initializer else None
        config = {
            'embed_size': self.embed_size,
            'discrete': self.discrete,
            'n_symbols': self.n_symbols,
            'dropout': self.dropout_rate,
            'batch_norm': self.use_batch_norm,
            'n_embed_layers': self.n_embed_layers,
            'embedding_initializer': ei,
            'freeze_embeddings': self.freeze_embeddings,
            'use_position_encoding': self.use_position_encoding,
            'concat_position_encoding': self.concat_position_encoding,
            'reproject_position_encoding': self.reproject_position_encoding,
            'kernel_regularizer':
            tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
            tf.keras.regularizers.serialize(self.activity_regularizer)

        }
        return config

    @classmethod
    def from_config(cls, config):
        ei = config.pop('embedding_initializer')
        if ei:
            ei = np.array(ei)
        return cls(embedding_initializer=ei, **config)
