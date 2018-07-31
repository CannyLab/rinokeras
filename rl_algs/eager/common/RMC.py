import tensorflow as tf
import numpy as np
from rl_algs.eager.common.attention import MultiHeadAttentionMap, ScaledDotProductSimilarity


# https://arxiv.org/pdf/1806.01822.pdf
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
class RelationalMemoryCoreCell(tf.keras.Model):

    def __init__(self, mem_slots, head_size, num_heads=1, forget_bias=1.0, **kwargs):
        """Relational Memory Core Cell Constructor
            Args:
                mem_slots: The total number of memory slots to use.
                head_size: The size of an attention head.
                num_heads: The number of attention heads to use. Defaults to 1.
            Raises:
                ValueError: num_blocks is < 1.
                ValueError: attention_mlp_layers is < 1.
        """
        super().__init__(**kwargs)

        # Handle the memory size parameters
        self._mem_slots = mem_slots
        self._head_size = head_size
        self._num_heads = num_heads
        self._mem_size = self._head_size * self._num_heads

        self._forget_bias = forget_bias

        # Setup the layers which we'll use for the network
        self.g_phi = tf.keras.layers.Dense(self._mem_size)
        self.input_embedding = tf.keras.layers.Dense(self._mem_size)
        self.similarity_metric = ScaledDotProductSimilarity()
        self.attention_layer = MultiHeadAttentionMap(
            self.similarity_metric, self._num_heads, self._mem_size)

        # Setup the weights for the input layers
        self.w_f = tf.keras.layers.Dense(self._mem_size, use_bias=False)
        self.u_f = tf.keras.layers.Dense(self._mem_size)

        self.w_i = tf.keras.layers.Dense(self._mem_size, use_bias=False)
        self.u_i = tf.keras.layers.Dense(self._mem_size)

        self.w_o = tf.keras.layers.Dense(self._mem_size, use_bias=False)
        self.u_o = tf.keras.layers.Dense(self._mem_size)

    def call(self, inputs, states):
        h_state, m_state = states

        # h_state : [batch_size x mem_slots x mem_size]
        # m_state : [batch_size x mem_slots x mem_size]

        # Compute the proposal for the memory
        dense_inputs = self.input_embedding(inputs)
        memory_proposal = self.attention_layer((m_state,                                                    # Queries
                                                tf.contrib.keras.layers.concatenate([m_state,
                                                                                     tf.expand_dims(dense_inputs, axis=1)], axis=1),  # Keys
                                                tf.contrib.keras.layers.concatenate([m_state,
                                                                                     tf.expand_dims(dense_inputs, axis=1)], axis=1)),  # Values
                                               )
        memory_proposal = m_state + memory_proposal  # Residual connection
        memory_proposal = self.g_phi(
            memory_proposal) + memory_proposal  # Second residual block

        # Compute the gate updates for the RMC
        # the parameters are shared for each m_i, so h_state has shape self._mem_size
        # Generate the forget gate.
        forget_gate = tf.keras.activations.sigmoid(
            self.u_f(h_state) + np.expand_dims(self.w_f(inputs), axis=1))
        # Generate the input gate.
        input_gate = tf.keras.activations.sigmoid(
            self.u_i(h_state) + np.expand_dims(self.w_i(inputs), axis=1))
        # Generate the output gate.
        output_gate = tf.keras.activations.sigmoid(
            self.u_o(h_state) + np.expand_dims(self.w_o(inputs), axis=1))

        # Generate the updated memories.
        new_memories = tf.keras.activations.sigmoid(
            m_state * forget_gate + self._forget_bias) + memory_proposal * tf.keras.activations.sigmoid(input_gate)
        new_hidden = tf.keras.activations.tanh(
            m_state) * tf.keras.activations.sigmoid(output_gate)

        # Return the values
        return new_memories, (new_hidden, new_memories)
