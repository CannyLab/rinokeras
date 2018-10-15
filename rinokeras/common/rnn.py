import tensorflow as tf

from .attention import LuongAttention

# LSTM Code from https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/06_02_custom_rnn.ipynb


class EagerLSTMCell(tf.keras.Model):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

        def bias_initializer(_, *args, **kwargs):
            # Unit forget bias from the paper
            # - [Learning to forget: Continual prediction with LSTM]
            return tf.keras.backend.concatenate([
                tf.keras.initializers.Zeros()((self.units,), *args, **kwargs),  # input gate
                tf.keras.initializers.Ones()((self.units,), *args, **kwargs),  # forget gate
                tf.keras.initializers.Zeros()((self.units * 2,), *args, **
                                              kwargs)  # context and output gates
            ])
        self.kernel = tf.keras.layers.Dense(4 * units, use_bias=False)
        self.recurrent_kernel = tf.keras.layers.Dense(4 * units,
                                                      kernel_initializer='glorot_uniform',
                                                      bias_initializer=bias_initializer)

    def call(self, inputs, states, training=None, mask=None):
        h_state, c_state = states
        # LSTM gate steps
        z = self.kernel(inputs) + self.recurrent_kernel(h_state)

        z0 = z[:, :self.units]
        z1 = z[:, self.units:2 * self.units]
        z2 = z[:, 2 * self.units:3 * self.units]
        z3 = z[:, 3 * self.units:]

        # gate updates
        i = tf.keras.activations.sigmoid(z0)
        f = tf.keras.activations.sigmoid(z1)
        c = f * c_state + i * tf.nn.tanh(z2)

        # state updates
        o = tf.keras.activations.sigmoid(z3)
        h = o * tf.nn.tanh(c)

        h_state = h
        c_state = c

        return h_state, (h_state, c_state)


class EagerLSTM(tf.keras.Model):
    def __init__(self, units, return_sequences=False, return_state=False, return_all_states=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.return_all_states = return_all_states
        self.cell = EagerLSTMCell(units)

    def call(self, inputs, training=None, mask=None, initial_state=None):
        # LSTM Cell in pure TF Eager code
        # reset the states initially if not provided, else use those
        if initial_state is None:
            h_state = tf.zeros((inputs.shape[0], self.units))
            c_state = tf.zeros((inputs.shape[0], self.units))
        else:
            assert len(
                initial_state) == 2, "Must pass a list of 2 states when passing 'initial_state'"
            h_state, c_state = initial_state

        h_list = []
        c_list = []

        state = (h_state, c_state)
        for t in range(inputs.shape[1]):
            ip = inputs[:, t]
            output, state = self.cell(ip, states=state)
            h_state, c_state = state

            h_list.append(h_state)
            c_list.append(c_state)

        hidden_outputs = tf.stack(h_list, axis=1)
        hidden_states = tf.stack(c_list, axis=1)

        if self.return_all_states:
            return hidden_outputs, hidden_outputs, hidden_states
        if self.return_state and self.return_sequences:
            return hidden_outputs, h_state, c_state
        elif self.return_state and not self.return_sequences:
            return h_state, h_state, c_state
        elif self.return_sequences and not self.return_state:
            return hidden_outputs
        else:
            return h_state


class EagerBidirectionalLSTM(tf.keras.Model):

    def __init__(self, units, return_sequences=False, return_state=False, **kwargs):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.forward_lstm = EagerLSTM(units, return_sequences, return_state)
        self.backward_lstm = EagerLSTM(units, return_sequences, return_state)

    def call(self, inputs, training=None, mask=None):

        reverse_inputs = inputs[:, ::-1]

        res1 = self.forward_lstm(inputs)
        res2 = self.backward_lstm(reverse_inputs)
        return res1, res2


class FixedLengthDecoder(tf.keras.Model):

    def __init__(self, units, output_size, output_layer=None, attention=None):
        super().__init__()
        self.cell = EagerLSTMCell(units)
        self.output_size = output_size
        self.attention = attention
        if attention == 'luong':
            self.attention_fn = LuongAttention()
        elif attention == 'luonglocal':
            self.attention_fn = LuongAttention(local=True)

        if output_layer is None:
            self.output_layer = tf.keras.layers.Dense(output_size)
        else:
            self.output_layer = output_layer

        self.target_embedding = tf.keras.layers.Dense(units, activation='relu')

    def call(self, inputs, seq_len, target_inputs=None):
        if self.attention:
            initial_state, source_sequence = inputs
        else:
            initial_state = inputs
        batch_size = initial_state[0].shape[0]

        if target_inputs is None:
            inputs = tf.zeros((batch_size, self.output_size))
        elif isinstance(target_inputs, list):
            inputs = target_inputs[0]
        else:
            inputs = target_inputs[:, 0]

        outputs = []
        state = initial_state
        for t in range(seq_len):
            inputs = self.target_embedding(inputs)
            output, state = self.cell(inputs, states=state)
            if self.attention:
                output = self.attention_fn((output, source_sequence), t=t)
            output = self.output_layer(output)
            outputs.append(output)

            if target_inputs is None:
                inputs = output
            elif isinstance(target_inputs, list):
                inputs = target_inputs[t + 1]
            else:
                inputs = target_inputs[:, t + 1]
        outputs = tf.stack(outputs, 1)
        return outputs


__all__ = ['EagerLSTMCell', 'EagerLSTM', 'EagerBidirectionalLSTM', 'FixedLengthDecoder']
