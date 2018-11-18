
import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer


class GraphConvolutionalLayer(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConvolutionalLayer, self).__init__(activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=3)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)  # The input should be [batch_size x units x N]
        if input_shape[-1].value is None or input_shape[-2].value is None:
            raise ValueError('The last two dimensions of the GCN are wrong: {}. Input should be [BS x Units x N]'.format(input_shape))
        self.input_spec = InputSpec(min_ndim=3,
                                    axes={-1: input_shape[-1].value, -2: input_shape[-2].value})

        self.kernel = self.add_weight(
            'kernel',
            shape=[self.units, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)

    def call(self, inputs, adj_matrix):

        # Inputs is n x d_model, so we need to change it
        inputs = tf.transpose(inputs, [0, 2, 1])

        kernel_mat = tf.ones([tf.shape(inputs)[0], 1, 1], dtype=self.dtype) * self.kernel
        outputs = tf.matmul(kernel_mat, inputs)
        degree_mat = tf.linalg.diag(1.0 / tf.reduce_sum(adj_matrix, axis=1))
        outputs = tf.matmul(outputs, adj_matrix + tf.eye(tf.shape(adj_matrix)[-1], dtype=self.dtype))
        outputs = tf.matmul(outputs,degree_mat)

        outputs = tf.transpose(outputs, [0, 2, 1])

        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
