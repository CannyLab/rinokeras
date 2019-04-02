from typing import Optional
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout
import warnings

import rinokeras.core.v1x as rk
from rinokeras.core.v1x.common.layers import InvertibleDense, CouplingLayer
from rinokeras.core.v1x.common.layers import WeightNormDense as Dense

from rinokeras.core.v1x.models.transformer import TransformerInputEmbedding, TransformerEncoder, TransformerDecoderBlock
from rinokeras.core.v1x.train import Experiment


class EvenOddInvertibleDense(Model):

    def __init__(self, out_size):
        super().__init__()
        warnings.warn('GLOW code is untested -- USE AT YOUR OWN RISK.', FutureWarning)
        self.invertible_dense = InvertibleDense(2 * out_size)

    def call(self, inputs, reverse=False):
        batch_size = tf.shape(inputs)[0]
        sequence_length = inputs.shape[1] if inputs.shape[1].value is not None \
            else tf.shape(inputs)[1]

        inputs = tf.reshape(inputs, (batch_size, sequence_length // 2, 2 * inputs.shape[-1]))

        if reverse:
            outputs = self.invertible_dense(inputs, reverse=reverse)
        else:
            outputs, log_det_W = self.invertible_dense(inputs, reverse=reverse)

        outputs = tf.reshape(outputs, (batch_size, sequence_length, inputs.shape[-1] // 2))

        return outputs if reverse else (outputs, log_det_W)


class EvenOddCouplingLayer(Model):

    def __init__(self,
                 out_size: int,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float] = None,
                 layer_dropout: Optional[float] = None,
                 kernel_regularizer=None) -> None:
        super().__init__()
        warnings.warn('GLOW code is untested -- USE AT YOUR OWN RISK.', FutureWarning)
        self.out_size = out_size
        self.project = Dense(hidden_size, activation='relu')
        self.decoder_block = TransformerDecoderBlock(
            n_heads, filter_size, hidden_size, dropout, layer_dropout, kernel_regularizer=kernel_regularizer)
        self.pred_s = Dense(out_size, kernel_initializer='zeros', bias_initializer='zeros')
        self.pred_t = Dense(out_size, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, inputs, encoder_outputs, reverse=False, encoder_mask=None, decoder_mask=None):
        """
        Args:
            inputs - Tensor with shape [batch_size, sequence_length, out_size]
            encoder_output - Tensor with shape [batch_size, sequence_length, d_model]

        Returns:
            output, log_s
        """

        batch_size = tf.shape(encoder_outputs)[0]
        sequence_length = encoder_outputs.shape[1] if encoder_outputs.shape[1].value is not None \
            else tf.shape(encoder_outputs)[1]

        even_inputs = inputs[:, ::2]
        odd_inputs = inputs[:, 1::2]

        self_attention_mask = decoder_mask[:, ::2, ::2] if decoder_mask is not None else None
        cross_attention_mask = self.get_cross_attention_mask(
            encoder_outputs, even_inputs, encoder_mask, self_attention_mask)

        projected_inputs = self.project(even_inputs)
        transform = self.decoder_block(
            projected_inputs, encoder_outputs=encoder_outputs,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask)

        log_s = self.pred_s(transform)
        t = self.pred_t(transform)

        if reverse:
            odd_transform = (odd_inputs - t) / tf.exp(log_s)
        else:
            odd_transform = tf.exp(log_s) * odd_inputs + t

        outputs = tf.stack((even_inputs, odd_transform), 2)
        outputs = tf.reshape(outputs, (batch_size, sequence_length, self.out_size))

        return outputs if reverse else (outputs, log_s)

    def get_cross_attention_mask(self, encoder_output, decoder_input, encoder_mask, decoder_mask):
        if encoder_mask is None and decoder_mask is None:
            cross_attention_mask = None
        elif encoder_mask is None:
            # We need to not mask the encoding, but mask the decoding
            # The decoding mask should have shape [batch_size x target_len x target_len]
            # meaning all we have to do is pad the mask out properly
            cross_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                                (1, tf.shape(encoder_output)[1], 1)), (0, 2, 1))
        elif decoder_mask is None:
            cross_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                                (1, 1, tf.shape(decoder_input)[1])), (0, 2, 1))
        else:
            dec_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                              (1, tf.shape(encoder_output)[1], 1)), (0, 2, 1))
            enc_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                              (1, 1, tf.shape(decoder_input)[1])), (0, 2, 1))
            cross_attention_mask = tf.logical_and(enc_attention_mask, dec_attention_mask)

        return cross_attention_mask


class TransformerGlowModel(Model):

    def __init__(self,
                 discrete: bool = True,
                 n_symbols: Optional[int] = None,
                 out_size: int = -1,
                 output_activation: Optional[str] = None,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_filter: int = 2048,
                 n_flows: int = 6,
                 dropout: Optional[float] = None,
                 layer_dropout: Optional[float] = None,
                 embedding_initializer=None,
                 use_preembedded_vectors=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        assert out_size > 0
        super().__init__()
        warnings.warn('GLOW code is untested -- USE AT YOUR OWN RISK.', FutureWarning)
        self.discrete = discrete
        self.n_symbols = n_symbols
        self.out_size = out_size
        self.output_activation = output_activation
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_filter = d_filter
        self.n_flows = n_flows
        self.dropout = Dropout(0 if dropout is None else dropout)

        input_embedding = TransformerInputEmbedding(
            d_model, discrete, n_symbols, dropout, embedding_initializer=embedding_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)

        self.encoder = TransformerEncoder(
            input_embedding,
            n_layers, n_heads, d_model, d_filter, dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)

        self.invertible_dense = [EvenOddInvertibleDense(out_size) for _ in range(n_flows)]
        self.WN = [EvenOddCouplingLayer(out_size, n_heads, d_filter, d_model, dropout,
                                        layer_dropout, kernel_regularizer)
                   for _ in range(n_flows)]

    def call(self, inputs, source_mask=None, target_mask=None):
        source, target = inputs

        source_mask = rk.utils.convert_to_attention_mask(source, source_mask)
        target_mask = rk.utils.convert_to_attention_mask(target, target_mask)

        encoder_outputs = self.encoder(source, encoder_mask=source_mask)

        do_pad = tf.equal(tf.mod(tf.shape(target)[1], 2), 1)
        target = K.switch(
            do_pad, tf.pad(target, [[0, 0], [0, 1], [0, 0]]), target)
        encoder_outputs = K.switch(
            do_pad, tf.pad(encoder_outputs, [[0, 0], [0, 1], [0, 0]]), encoder_outputs)

        # also pad source mask, target_mask

        output = target
        log_det_W_list = []
        log_s_list = []
        for i in range(self.n_flows):
            output, log_det_W = self.invertible_dense[i](output)
            output, log_s = self.WN[i](
                output, encoder_outputs=encoder_outputs,
                encoder_mask=source_mask,
                decoder_mask=target_mask)
            log_s_list.append(log_s)
            log_det_W_list.append(log_det_W)

        return output, log_s_list, log_det_W_list

    def predict(self, inputs, source_mask=None, target_mask=None):
        source, output = inputs
        source_mask = rk.utils.convert_to_attention_mask(source, source_mask)
        target_mask = rk.utils.convert_to_attention_mask(output, target_mask)

        encoder_outputs = self.encoder(source, encoder_mask=source_mask)

        do_pad = tf.equal(tf.mod(tf.shape(encoder_outputs)[1], 2), 1)
        encoder_outputs = K.switch(
            do_pad, tf.pad(encoder_outputs, [[0, 0], [0, 1], [0, 0]]), encoder_outputs)

        # output = tf.random_normal((tf.shape(source)[0], tf.shape(source)[1], 2 * self.out_size))

        for k in reversed(range(self.n_flows)):
            output = self.WN[k](
                output, encoder_outputs=encoder_outputs, encoder_mask=source_mask,
                decoder_mask=target_mask, reverse=True)
            output = self.invertible_dense[k](output, reverse=True)

        return output

    def get_cross_attention_mask(self, encoder_output, decoder_input, encoder_mask, decoder_mask):
        if encoder_mask is None and decoder_mask is None:
            cross_attention_mask = None
        elif encoder_mask is None:
            # We need to not mask the encoding, but mask the decoding
            # The decoding mask should have shape [batch_size x target_len x target_len]
            # meaning all we have to do is pad the mask out properly
            cross_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                                (1, tf.shape(encoder_output)[1], 1)), (0, 2, 1))
        elif decoder_mask is None:
            cross_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                                (1, 1, tf.shape(decoder_input)[1])), (0, 2, 1))
        else:
            dec_attention_mask = tf.transpose(tf.tile(decoder_mask[:, 1, :][:, None, :],
                                              (1, tf.shape(encoder_output)[1], 1)), (0, 2, 1))
            enc_attention_mask = tf.transpose(tf.tile(encoder_mask[:, 1, :][:, :, None],
                                              (1, 1, tf.shape(decoder_input)[1])), (0, 2, 1))
            cross_attention_mask = tf.logical_and(enc_attention_mask, dec_attention_mask)

        return cross_attention_mask


class TransformerGlowExperiment(Experiment):

    def __init__(self, sigma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        warnings.warn('GLOW code is untested -- USE AT YOUR OWN RISK.', FutureWarning)

    def loss_function(self, inputs, outputs):
        z, log_s_list, log_det_W_list = outputs

        log_s_total = sum(tf.reduce_sum(log_s) for log_s in log_s_list)
        log_det_W_total = sum(log_det_W_list)

        loss = tf.reduce_sum(tf.square(z)) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total

        return loss / tf.cast(tf.size(loss), tf.float32)
