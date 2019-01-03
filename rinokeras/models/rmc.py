from typing import Optional
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Flatten, RNN

from .transformer import TransformerMultiAttention, TransformerFeedForward, TransformerEncoderBlock
from rinokeras.common.layers import Highway
from rinokeras.common.layers import WeightNormDense as Dense


# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
class RelationalMemoryCoreCell(Model):

    def __init__(self,
                 mem_slots: int,
                 mem_size: int,
                 n_heads: int,
                 forget_bias: float = 1.0,
                 input_bias: float = 0.0,
                 dropout: Optional[float] = None,
                 gate_style: str = 'unit',
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
        self.state_size = 128

        self.initial_embed = Dense(mem_size, use_bias=True)

        self.attend_over_memory = TransformerEncoderBlock(
            n_heads, 4 * mem_size, mem_size, dropout, 0.0,
            kernel_regularizer, bias_regularizer, activity_regularizer)

        self.flatten = Flatten()
        num_gates = self._calculate_gate_size() * 2
        self.gate_inputs = Dense(num_gates, use_bias=True)
        self.gate_memory = Dense(num_gates, use_bias=True)

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
        init_state = tf.one_hot(tf.range(self.mem_slots), self.mem_size, dtype=dtype)
        init_state = tf.tile(init_state[None], (batch_size, 1, 1))
        return init_state

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
        inputs = self.flatten(memory)
        gate_inputs = self.gate_inputs(inputs)
        gate_inputs = gate_inputs[:, None]
        gate_memory = self.gate_memory(memory)
        input_gate, forget_gate = tf.split(gate_memory + gate_inputs, num_or_size_splits=2, axis=-1)

        input_gate = tf.sigmoid(input_gate + self.input_bias)
        forget_gate = tf.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states, treat_input_as_matrix=False):
        """Runs the relational memory core.

        Args:
            inputs: Tensor input of shape [batch_size, n_dims]
            memory: Memory output from previous timestep
            treat_input_as_matrix: Optional, whether to treat the `input`
                as a sequence with the same shape as memory

        Returns:
            output: This time step's output
            next_memory: This time step's memory
        """
        memory = states[0]
        if treat_input_as_matrix:
            inputs.shape.assert_has_rank(3)
            inputs = self.initial_embed(inputs)
        else:
            inputs.shape.assert_has_rank(2)
            inputs = self.initial_embed(inputs)
            inputs = inputs[:, None]  # expand the first dimension so it will concat across mem slots

        memory_plus_input = K.concatenate((memory, inputs), axis=1)
        next_memory = self.attend_over_memory(memory_plus_input)
        next_memory = next_memory[:, :-tf.shape(inputs)[1], :]

        if self.gate_style == 'unit' or self.gate_style == 'memory':
            input_gate, forget_gate = self.create_gates(inputs, memory)
            next_memory = input_gate * tf.tanh(next_memory)
            next_memory += forget_gate * tf.tanh(memory)

        output = next_memory
        return output, next_memory


class RelationalMemoryCore(RNN):

    def __init__(self,
                 mem_slots: int,
                 mem_size: int,
                 n_heads: int,
                 forget_bias: float = 1.0,
                 input_bias: float = 0.0,
                 dropout: Optional[float] = None,
                 gate_style: str = 'gate',
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
            mem_slots, mem_size, n_heads, forget_bias, input_bias, dropout,
            gate_style, kernel_regularizer, bias_regularizer, activity_regularizer)
        super().__init__(
            cell, return_sequences=return_sequences, return_state=return_state,
            go_backwards=go_backwards, stateful=stateful, unroll=unroll, **kwargs)
