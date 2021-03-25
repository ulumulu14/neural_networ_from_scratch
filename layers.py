import numpy as np
from abc import ABC
from abc import abstractmethod


class Layer(ABC):

    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, d_inputs):
        raise NotImplementedError()

    @abstractmethod
    def get_details(self):
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self, input_size, n_neurons, name=None, weight_regularizer_l1=0, bias_regularizer_l1=0,
                 weight_regularizer_l2=0, bias_regularizer_l2=0):
        if not isinstance(input_size, int):
            raise TypeError('Input size must be an integer')
        if not isinstance(n_neurons, int):
            raise TypeError('Number of neurons must be an integer')
        if n_neurons <= 0:
            raise ValueError('Number of neurons cant be less or equal to 0')
        if input_size <= 0:
            raise ValueError('Input size cant be less or equal to 0')
        if weight_regularizer_l1 < 0:
            raise ValueError('Weight regularizer l1 cant be less than 0')
        if bias_regularizer_l1 < 0:
            raise ValueError('Bias regularizer l1 cant be less than 0')
        if weight_regularizer_l2 < 0:
            raise ValueError('Weight regularizer l2 cant be less than 0')
        if bias_regularizer_l2 < 0:
            raise ValueError('Bias regularizer l2 cant be less than 0')

        super(Dense, self).__init__(name)
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.weights = 0.1 * np.random.randn(input_size, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self._inputs = None
        self._d_inputs = None
        self._d_weights = None
        self._d_biases = None
        # For SGD with momentum
        self.weights_momentums = None
        self.biases_momentums = None
        # For AdaGrad, RMSProp, holds previous gradients squared
        self.d_weights_history = None
        self.d_biases_history = None
        # Regularization parameters
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2

    @property
    def inputs(self):
        if self._inputs is None:
            raise ValueError('inputs is None')

        return self._inputs

    @property
    def d_inputs(self):
        if self._d_inputs is None:
            raise ValueError('d_inputs is None')
        return self._d_inputs

    @property
    def d_weights(self):
        if self._d_weights is None:
            raise ValueError('d_weights is None')

        return self._d_weights

    @property
    def d_biases(self):
        if self._d_biases is None:
            raise ValueError('d_biases is None')

        return self._d_biases

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @d_inputs.setter
    def d_inputs(self, d_inputs):
        self._d_inputs = d_inputs

    @d_weights.setter
    def d_weights(self, d_weights):
        self._d_weights = d_weights

    @d_biases.setter
    def d_biases(self, d_biases):
        self._d_biases = d_biases

    def forward(self, inputs):
        self.inputs = np.array(inputs)

        return np.dot(self.inputs, self.weights) + self.biases

    def backward(self, gradient):
        # gradient argument is gradient of next layer

        # Gradient w.r.t weights and biases
        self.d_weights = np.dot(self.inputs.T, gradient)
        self.d_biases = np.sum(gradient, axis=0, keepdims=True)

        # Gradients on regularization
        if self.weight_regularizer_l1 > 0:
            d_l1 = np.ones_like(self.weights)
            d_l1[self.weights < 0] = -1
            self._d_weights += self.weight_regularizer_l1 * d_l1

        if self.bias_regularizer_l1 > 0:
            d_l1 = np.ones_like(self.biases)
            d_l1[self.biases < 0] = -1
            self.d_biases += self.bias_regularizer_l1 * d_l1

        if self.weight_regularizer_l2 > 0:
            self.d_weights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l2 > 0:
            self.d_biases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient w.r.t inputs
        self.d_inputs = np.dot(gradient, self.weights.T)

        return self.d_inputs

    def get_details(self):
        return f'Name: {self.name} || Type: Dense || Output Size: {self.n_neurons}\n'


class Dropout(Layer):

    def __init__(self, rate, name=None):
        if 0 > rate > 1:
            raise ValueError('Rate value must be (0, 1)')

        super(Dropout, self).__init__(name)
        self._rate = 1 - rate
        self._inputs = None
        self._binary_mask = None

    def forward(self, inputs):
        self._inputs = inputs
        self._binary_mask = np.random.binomial(1, self._rate, size=self._inputs.shape) / self._rate

        return self._inputs * self._binary_mask

    def backward(self, d_inputs):
        return d_inputs * self._binary_mask

    def get_details(self):
        return f'Name: {self.name} || Type: Dense || Output Size: {self._inputs.shape}\n'
