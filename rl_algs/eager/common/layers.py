import collections
import tensorflow as tf

class RandomNoise(tf.keras.layers.Layer):

    def __init__(self, shape, initial):
        super().__init__()
        self._shape = shape
        self._logstd = self.add_variable('logstd', shape, dtype=tf.float32,
                                            initializer=tf.constant_initializer(initial))
    def call(self, inputs):
        epsilon = tf.random_normal(self._shape)
        return inputs + epsilon * tf.exp(self._logstd)

    @property
    def logstd(self):
        return self._logstd

    @property
    def std(self):
        return tf.exp(self._logstd)

class LuongAttention(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, inputs):
        self.attention_weights = self.add_variable('attention_weights', (inputs[0][-1] + inputs[1][-1], inputs[1][-1]),
                                                                        initializer=tf.initializers.variance_scaling())

    def call(self, inputs):
        target_hidden, source_hidden_sequence = inputs
        # source hidden sequence shape -> (None, None, encoder_cell_size)
        # target hidden shape -> (None, decoder_cell_size)
        score = tf.matmul(source_hidden_sequence, tf.expand_dims(target_hidden, -1))
        alignment = tf.nn.softmax(score, 1)
        weighted = tf.reduce_sum(source_hidden_sequence * alignment, 1) # will broadcast over third dimension
        concatenated = tf.concat((target_hidden, weighted), 1)
        output = tf.tanh(tf.matmul(concatenated, self.attention_weights))
        return output

class FixedLengthDecoder(tf.keras.Model):

    def __init__(self, units, attention=None):
        super().__init__()
        self.cell = tf.keras.layers.LSTMCell(units)
        self.attention = attention
        if attention == 'luong':
            self.attention_fn = LuongAttention()

    def call(self, inputs, seq_len, target_inputs=None):
        if self.attention:
            initial_state, source_sequence = inputs
        else:
            initial_state = inputs
        batch_size = initial_state[0].shape[0]
        if target_inputs is None:
            inputs = tf.zeros((batch_size, self.cell.units))
        elif isinstance(target_inputs, list):
            inputs = target_inputs[0]
        else:
            inputs = target_inputs[:,0]

        outputs = []
        state = initial_state
        for t in range(seq_len):
            output, state = self.cell(inputs, states=state)

            if self.attention:
                output = self.attention_fn((output, source_sequence))
            outputs.append(output)

            if target_inputs is None:
                inputs = output
            elif isinstance(target_inputs, list):
                inputs = target_inputs[t + 1]
            else:
                inputs = target_inputs[:,t + 1]
        outputs = tf.stack(outputs, 1)
        return outputs

# I presume this is just how Sequential is added but at the moment Sequential requires input size to be specified at the begining
class Stack(tf.keras.Model):
    def __init__(self, layers=None):
        super().__init__()
        self._call = None
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)

    def call(self, inputs, training=None, mask=None):
        output = inputs
        for layer in self._layers:
            output = layer(output)
        return output

class Conv2DStack(Stack):

    def __init__(self, layers, batch_norm=False, activation='relu', flatten_output=True):
        super().__init__()
        if layers is None:
            layers = []
        def cast_layer(inputs):
            if inputs.dtype == tf.uint8:
                return tf.cast(inputs, tf.float32) / 255
        self.add(cast_layer)
        for layer in layers:
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(tf.keras.layers.Conv2D(*layer))
            if batch_norm:
                self.add(tf.keras.layers.BatchNormalization())
            self.add(tf.keras.layers.Activation(activation))
        self.add(tf.keras.layers.Flatten())

class DenseStack(Stack):

    def __init__(self, layers, activation='relu'):
        super().__init__()
        if layers is None:
            layers = []
        for i, layer in enumerate(layers):
            if not isinstance(layer, collections.Iterable):
                layer = (layer,)
            self.add(tf.keras.layers.Dense(*layer))
            self.add(tf.keras.layers.Activation(activation))

# class SimpleNN(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.all_layers = [tf.keras.layers.Dense(10, activation='relu')]
#         self.all_layers.append(tf.keras.layers.Dense(2))

#     def call(self, input_data):
#         output = input_data
#         for layer in self.all_layers:
#             output = layer(output)
#         return output

# class MNISTModel(tf.keras.Model):

#     def __init__(self):
#         super().__init__()
#         self.conv = Conv2DStack([(16, 5, 2), (32, 5, 2)])
#         self.classify = tf.keras.layers.Dense(10)

#     def call(self, inputs):
#         net = self.conv(inputs)
#         logits = self.classify(net)
#         output = tf.nn.softmax(logits)
#         return output
