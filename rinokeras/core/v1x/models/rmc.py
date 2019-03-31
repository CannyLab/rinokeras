from typing import Optional

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import RNN, Flatten, Reshape, Input, LSTMCell
import numpy as np

import rinokeras.core.v1x as rk
from rinokeras.core.v1x.common.layers import WeightNormDense as Dense
from rinokeras.core.v1x.common.attention import AttentionMap, ScaledDotProductSimilarity, AttentionQKVProjection
from rinokeras.core.v1x.common.layers import PositionEmbedding, LearnedEmbedding, LayerDropout, \
    LayerNorm, DenseStack, Dropout

from .transformer import TransformerEncoderBlock, TransformerMultiAttention, TransformerFeedForward


class RMCFeedForward(Model):

    def __init__(self,
                 mem_slots: int,
                 filter_size: int,
                 hidden_size: int,
                 dropout: Optional[float],
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        self.mem_slots = mem_slots
        self.ff_layers = [
            TransformerFeedForward(
                filter_size, hidden_size, dropout,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer)
            for _ in range(mem_slots)]

    def call(self, inputs):
        inputs = tf.split(inputs, axis=1, num_or_size_splits=self.mem_slots)
        outputs = [ff(inp) for ff, inp in zip(self.ff_layers, inputs)]
        return tf.concat(outputs, 1)


class RMCBlock(Model):
    """A decoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: two Tensors encoder_outputs, decoder_inputs
                    encoder_outputs -> a Tensor with shape [batch_size, sequence_length, channels]
                    decoder_inputs -> a Tensor with shape [batch_size, decoding_sequence_length, channels]

    :return: output: Tensor with same shape as decoder_inputs
    """

    def __init__(self,
                 mem_slots: int,
                 n_heads: int,
                 filter_size: int,
                 hidden_size: int,
                 key_size: Optional[int] = None,
                 dropout: Optional[float] = None,
                 layer_dropout: Optional[float] = None,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None) -> None:
        super().__init__()
        self.mem_slots = mem_slots
        self.n_heads = n_heads
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

        self.multi_attention = TransformerMultiAttention(
            n_heads, dropout,
            key_size=key_size,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        self.layer_drop_1 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)
        self.self_attention = TransformerMultiAttention(
            n_heads, dropout,
            key_size=key_size,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        self.layer_drop_2 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)
        # self.feed_forward = RMCFeedForward(mem_slots, filter_size, hidden_size, dropout,
                                           # kernel_initializer=kernel_initializer,
                                           # kernel_regularizer=kernel_regularizer,
                                           # bias_regularizer=bias_regularizer,
                                           # activity_regularizer=activity_regularizer)
        self.feed_forward = TransformerFeedForward(filter_size, hidden_size, 
                                                   dropout=dropout,
                                                   kernel_initializer=kernel_initializer,
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer)
        self.layer_drop_3 = LayerDropout(
            0 if layer_dropout is None else layer_dropout)

    def call(self,
             memory_cells,
             rmc_inputs,
             cross_attention_mask=None,
             return_self_attention_weights=False,
             return_cross_attention_weights=False):
        # The cross-attention mask should have shape [batch_size x target_len x input_len]

        # Compute the attention using the keys/values from the encoder, and the query from the
        # decoder. This takes the encoder output of size [batch_size x source_len x d_model] and the
        # target self-attention layer of size [batch_size x target_len x d_model] and then computes
        # a multi-headed attention across them, giving an output of [batch_size x target_len x d_model]
        # using the encoder as the keys and values and the target as the queries
        memory_cells_cross, cross_attention_weights = self.multi_attention(
            (rmc_inputs, memory_cells),
            mask=cross_attention_mask,
            return_attention_weights=True)
        memory_cells_cross = self.layer_drop_1(memory_cells_cross, memory_cells)

        # Compute the selt-attention over the decoder inputs. This uses the self-attention
        # mask to control for the future outputs.
        # This generates a tensor of size [batch_size x target_len x d_model]
        memory_cells_self, self_attention_weights = self.self_attention(
            memory_cells_cross,
            source=memory_cells,
            mask=None,
            return_attention_weights=True)
        memory_cells_self = self.layer_drop_2(memory_cells_self, memory_cells_cross)

        output = self.feed_forward(memory_cells_self)
        output = self.layer_drop_3(output, memory_cells_self)

        if not (return_self_attention_weights or return_cross_attention_weights):
            return output
        elif return_self_attention_weights and not return_cross_attention_weights:
            return output, self_attention_weights
        elif not return_self_attention_weights and return_cross_attention_weights:
            return output, cross_attention_weights
        elif return_self_attention_weights and return_cross_attention_weights:
            return output, self_attention_weights, cross_attention_weights


# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
class RelationalMemoryCoreCell(Model):

    def __init__(self,
                 mem_slots: int,
                 mem_size: int,
                 n_heads: int,
                 key_size: Optional[int] = None,
                 forget_bias: float = 1.0,
                 input_bias: float = 0.0,
                 dropout: Optional[float] = None,
                 layer_norm: bool = False,
                 gate_style: str = 'unit',
                 treat_input_as_sequence: bool = False,
                 use_cross_attention: bool = False,
                 return_attention_weights: bool = False,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.n_heads = n_heads
        self.input_bias = input_bias
        self.forget_bias = forget_bias
        self.gate_style = gate_style
        self.state_size = (mem_slots * mem_size,)
        self.treat_input_as_sequence = treat_input_as_sequence
        self.use_cross_attention = use_cross_attention
        self.return_attention_weights = return_attention_weights
        self.layer_norm = layer_norm

        self.reshape = Reshape((mem_slots, mem_size))
        self.initial_embed = Dense(
            mem_size, activation='relu', use_bias=True,
            kernel_initializer=kernel_initializer)

        if use_cross_attention:
            self.attend_over_memory = RMCBlock(
                mem_slots,
                n_heads, mem_size * 4, mem_size,
                key_size=key_size, dropout=dropout,
                layer_dropout=None, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                kernel_initializer=kernel_initializer)
        else:
            self.attend_over_memory = TransformerEncoderBlock(
                n_heads, mem_size * 4, mem_size, 
                dropout=dropout,
                layer_dropout=None, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                kernel_initializer=kernel_initializer)

        if self.gate_style == 'attention':
            self.attention_map = AttentionMap(ScaledDotProductSimilarity())  # ,tf.identity
            self.qkv_projection = AttentionQKVProjection(
                self.mem_size, self.mem_size, project_value=True, kernel_initializer=kernel_initializer)
        if treat_input_as_sequence:
            self.similarity = ScaledDotProductSimilarity()

        self.posembed = PositionEmbedding()
        self.flatten = Flatten()
        num_gates = self._calculate_gate_size() * 2
        self.gate_inputs = Dense(
            num_gates, use_bias=True, kernel_initializer=kernel_initializer)
        self.gate_memory = Dense(
            num_gates, use_bias=True, kernel_initializer=kernel_initializer)
        self.memory_projection = Dense(
            16, use_bias=False, kernel_initializer=kernel_initializer)
        self.input_projection = Dense(
            16, use_bias=False, kernel_initializer=kernel_initializer)
        if layer_norm:
            self.norm = LayerNorm()
        self._initial_state = None
        if not tf.executing_eagerly():
            self._batch_size_ph = tf.placeholder(tf.int32, shape=[])

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Creates the initial memory.

        We should ensure each row of the memory is initialized to be unique,
        so initialize the matrix to be the identity. We then pad or truncate
        as necessary so that init_state is of size
        (batch_size, self._mem_slots, self._mem_size).

        Args:
            batch_size: The size of the batch.

        Returns:
            init_state: A truncated or padded matrix of size
                (batch_size, self._mem_slots, self._mem_size).
        """

        if batch_size is None:
            assert inputs is not None, 'Must pass either batch_size or inputs'
            batch_size = tf.shape(inputs)[0]

        if dtype is None:
            dtype = tf.float32
        zeros = tf.zeros((batch_size, self.mem_slots, self.mem_size), dtype=dtype)
        position = self.posembed(zeros)
        return [self.flatten(position)]
        # init_state = tf.one_hot(tf.range(self.mem_slots),
                                # self.mem_size, dtype=dtype)
        # init_state = tf.tile(init_state[None], (batch_size, 1, 1))
        # init_state = self.flatten(init_state)

        # return init_state

    def get_initial_state_numpy(self, batch_size: int):
        if not tf.executing_eagerly():
            if self._initial_state is None:
                self._initial_state = self.get_initial_state(batch_size=self._batch_size_ph)

            sess = K.get_session()
            return sess.run(self._initial_state, feed_dict={self._batch_size_ph: batch_size})
        else:
            return [self.get_initial_state(batch_size=batch_size)[0].numpy()]

    def _calculate_gate_size(self):
        """Calculate the gate size from the gate_style.

        Returns:
            The per sample, per head parameter size of each gate.
        """

        if self.gate_style == 'unit':
            return self.mem_size
        elif self.gate_style == 'memory':
            return 1
        else:  # self._gate_style == None
            return 0

    def create_gates(self, inputs, memory):
        """Create input and forget gates for this step using `inputs` and `memory`.

        Args:
            inputs: Tensor input.

            memory: The current state of memory.
        Returns:
            input_gate: A LSTM-like insert gate.
            forget_gate: A LSTM-like forget gate.
        """
        memory = tf.tanh(memory)

        if not self.treat_input_as_sequence:
            inputs = self.flatten(inputs)
            gate_inputs = self.gate_inputs(inputs)
            gate_inputs = gate_inputs[:, None]
        else:
            # [batch_size, input_size, d_model]
            gate_inputs = self.gate_inputs(inputs)
            memory_proj = self.memory_projection(memory)
            input_proj = self.input_projection(inputs)

            # [batch_size, mem_cells, input_size]
            input_weights = tf.nn.softmax(self.similarity(memory_proj, input_proj), -1)
            gate_inputs = tf.matmul(input_weights, gate_inputs)

        gate_memory = self.gate_memory(memory)
        input_gate, forget_gate = tf.split(
            gate_memory + gate_inputs, num_or_size_splits=2, axis=-1)

        input_gate = tf.sigmoid(input_gate + self.input_bias)
        forget_gate = tf.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        """Runs the relational memory core.

        Args:
            inputs: Tensor input of shape [batch_size, n_dims]
            memory: Memory output from previous timestep

        Returns:
            output: This time step's output
            next_memory: This time step's memory
        """

        memory = states[0]
        memory = self.reshape(memory)

        if self.treat_input_as_sequence:
            inputs.shape.assert_has_rank(3)
            inputs = self.initial_embed(inputs)
        else:
            inputs.shape.assert_has_rank(2)
            inputs = self.initial_embed(inputs)
            # expand the first dimension so it will concat across mem slots
            inputs = inputs[:, None]

        if self.use_cross_attention:
            inputs_mask = tf.reduce_any(tf.cast(inputs, tf.bool), -1)
            inputs_mask = rk.utils.convert_to_attention_mask(memory, inputs_mask)
            next_memory, attention_weights = self.attend_over_memory(
                memory, rmc_inputs=inputs, cross_attention_mask=None, return_cross_attention_weights=True)
        else:
            memory_plus_input = K.concatenate((memory, inputs), axis=1)
            inputs_mask = tf.reduce_any(tf.cast(memory_plus_input, tf.bool), -1)
            inputs_mask = rk.utils.convert_to_attention_mask(memory_plus_input, inputs_mask)
            next_memory, attention_weights = self.attend_over_memory(
                memory_plus_input, self_attention_mask=None, return_attention_weights=True)
            next_memory = next_memory[:, :self.mem_slots, :]
            attention_weights = attention_weights[:, :self.mem_slots]

        if self.gate_style == 'unit' or self.gate_style == 'memory':
            input_gate, forget_gate = self.create_gates(inputs, memory)
            next_memory = input_gate * tf.tanh(next_memory)
            next_memory += forget_gate * memory
        elif self.gate_style == 'attention':

            memory_update = tf.tanh(next_memory)  # This is the input of the memory

            # Do a QKV projection
            qkv_projection = self.qkv_projection((inputs, memory_update, memory_update))
            _, attention_weights = self.attention_map(qkv_projection)

            # Reduce max
            max_attention = tf.reduce_max(attention_weights, axis=-1)  # [bs, num_slots]

            # Convex combination
            next_memory = values * max_attention + memory * (1.0 - max_attention)

        next_memory = self.flatten(next_memory)
        attention_weights = self.flatten(attention_weights)

        output = next_memory
        if self.layer_norm:
            output = self.norm(output)
        output = output if not self.return_attention_weights else tf.concat((output, attention_weights), 1)

        return output, [next_memory]


class RelationalMemoryCore(RNN):

    def __init__(self,
                 mem_slots: int,
                 mem_size: int,
                 n_heads: int,
                 key_size: Optional[int] = None,
                 forget_bias: float = 1.0,
                 input_bias: float = 0.0,
                 dropout: Optional[float] = None,
                 layer_norm: bool = False,
                 gate_style: str = 'unit',
                 treat_input_as_sequence: bool = False,
                 use_cross_attention: bool = False,
                 return_attention_weights: bool = False,
                 kernel_initializer: Optional[tf.keras.initializers.Initializer] = 'glorot_uniform',
                 kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 go_backwards: bool = False,
                 stateful: bool = False,
                 unroll: bool = False,
                 **kwargs) -> None:
        cell = RelationalMemoryCoreCell(
            mem_slots=mem_slots,
            mem_size=mem_size,
            n_heads=n_heads,
            key_size=key_size,
            forget_bias=forget_bias,
            input_bias=input_bias,
            dropout=dropout,
            gate_style=gate_style,
            layer_norm=layer_norm,
            treat_input_as_sequence=treat_input_as_sequence,
            use_cross_attention=use_cross_attention,
            return_attention_weights=return_attention_weights,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        super().__init__(
            cell, return_sequences=return_sequences, return_state=return_state,
            go_backwards=go_backwards, stateful=stateful, unroll=unroll, **kwargs)

    def get_initial_state_numpy(self, batch_size: int):
        return self.cell.get_initial_state_numpy(batch_size)

    @property
    def mem_slots(self) -> int:
        return self.cell.mem_slots

    @property
    def mem_size(self) -> int:
        return self.cell.mem_size

    @property
    def n_heads(self) -> int:
        return self.cell.n_heads
