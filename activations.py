import numpy as np
import layers


class ReLU(layers.Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None
        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs

        return np.maximum(0, inputs)

    def backward(self, d_inputs):
        self.d_inputs = d_inputs.copy()

        #return self.d_inputs[self.inputs <= 0] = 0

    def get_details(self):
        return f'Name: {self.name} || Type: ReLU || Output Size: {len(self.inputs)}\n'


class Softmax(layers.Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None
        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, d_inputs):
        pass

    def get_details(self):
        return f'Name: {self.name} || Type: Softmax || Output Size: {len(self.inputs)}\n'


class Sigmoid(layers.Layer):

    def __init__(self):
        super().__init__()
        self.inputs = None
        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return 1 / (1 + np.exp(-inputs))

    def backward(self, d_inputs):
        self.d_inputs = d_inputs.copy()

    def get_details(self):
        return f'Name: {self.name} || Type: Sigmoid || Output Size: {len(self.inputs)}\n'