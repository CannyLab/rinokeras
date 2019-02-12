import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.eager import context

from tensorflow.keras.layers import RNN, LSTMCell
from tensorflow.keras import regularizers


class MaskedLSTMCell(LSTMCell):
    """Cell class for the LSTM layer.

    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
                Default: hyperbolic tangent (`tanh`).
                If you pass `None`, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
                for the recurrent step.
                Default: hard sigmoid (`hard_sigmoid`).
                If you pass `None`, no activation is applied
                (ie. "linear" activation: `a(x) = x`).x
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
                used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix,
                used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        unit_forget_bias: Boolean.
                If True, add 1 to the bias of the forget gate at initialization.
                Setting it to true will also force `bias_initializer="zeros"`.
                This is recommended in [Jozefowicz et
                    al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
                the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
                the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
                Mode 1 will structure its operations as a larger number of
                smaller dot products and additions, whereas mode 2 will
                batch them into fewer, larger operations. These modes will
                have different performance profiles on different hardware and
                for different applications.
    """

    def build(self, input_shape):
        super().build((input_shape[0], input_shape[1] - 1))

    def call(self, inputs, states, training=None):
        mask = inputs[:, -1:]
        inputs = inputs[:, :-1]
        states = tuple(state * (1 - mask) for state in states)
        return super().call(inputs, states, training=training)


class MaskedLSTM(RNN):
    """Masked Long Short-Term Memory layer - Hochreiter 1997.

     Note that this cell is not optimized for performance on GPU. Please use
    `tf.keras.layers.CuDNNLSTM` for better performance on GPU.

    Arguments:
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use.
                    Default: hyperbolic tangent (`tanh`).
                    If you pass `None`, no activation is applied
                    (ie. "linear" activation: `a(x) = x`).
            recurrent_activation: Activation function to use
                    for the recurrent step.
                    Default: hard sigmoid (`hard_sigmoid`).
                    If you pass `None`, no activation is applied
                    (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
                    used for the linear transformation of the inputs..
            recurrent_initializer: Initializer for the `recurrent_kernel`
                    weights matrix,
                    used for the linear transformation of the recurrent state..
            bias_initializer: Initializer for the bias vector.
            unit_forget_bias: Boolean.
                    If True, add 1 to the bias of the forget gate at initialization.
                    Setting it to true will also force `bias_initializer="zeros"`.
                    This is recommended in [Jozefowicz et
                        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
                    the `kernel` weights matrix.
            recurrent_regularizer: Regularizer function applied to
                    the `recurrent_kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                    the output of the layer (its "activation")..
            kernel_constraint: Constraint function applied to
                    the `kernel` weights matrix.
            recurrent_constraint: Constraint function applied to
                    the `recurrent_kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            dropout: Float between 0 and 1.
                    Fraction of the units to drop for
                    the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
                    Fraction of the units to drop for
                    the linear transformation of the recurrent state.
            implementation: Implementation mode, either 1 or 2.
                    Mode 1 will structure its operations as a larger number of
                    smaller dot products and additions, whereas mode 2 will
                    batch them into fewer, larger operations. These modes will
                    have different performance profiles on different hardware and
                    for different applications.
            return_sequences: Boolean. Whether to return the last output.
                    in the output sequence, or the full sequence.
            return_state: Boolean. Whether to return the last state
                    in addition to the output.
            go_backwards: Boolean (default False).
                    If True, process the input sequence backwards and return the
                    reversed sequence.
            stateful: Boolean (default False). If True, the last state
                    for each sample at index i in a batch will be used as initial
                    state for the sample of index i in the following batch.
            unroll: Boolean (default False).
                    If True, the network will be unrolled,
                    else a symbolic loop will be used.
                    Unrolling can speed-up a RNN,
                    although it tends to be more memory-intensive.
                    Unrolling is only suitable for short sequences.

    """

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            logging.warning('`implementation=0` has been deprecated, '
                            'and now defaults to `implementation=1`.'
                            'Please update your layer call.')
        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn('%s: Note that this layer is not optimized for performance. '
                         'Please use tf.keras.layers.CuDNNLSTM for better '
                         'performance on GPU.', self)
        cell = MaskedLSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation)
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super().call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation
