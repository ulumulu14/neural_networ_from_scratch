import numpy as np

class SGD:

    def __init__(self, learning_rate=1., decay=0, momentum=0):
        if learning_rate < 0.:
            raise ValueError('Learning rate cant be less than 0')
        if decay < 0.:
            raise ValueError('Decay cant be less than 0')
        if momentum < 0.:
            raise ValueError('Momentum cant be less than 0')

        self._learning_rate = float(learning_rate)
        self._current_learning_rate = float(learning_rate)
        self._decay = float(decay)
        self._momentum = float(momentum)
        self._iteration = 0

    def update_params(self, layer):
        # Momentum is parameters change from previous iteration * self._momentum
        if self._momentum:
            if layer.weights_momentums is None:
                layer.weights_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)

            weights_updates = self._momentum * layer.weights_momentums - self._current_learning_rate * layer.d_weights
            layer.weight_momentums = weights_updates
            biases_updates = self._momentum * layer.biases_momentums - self._current_learning_rate * layer.d_biases
            layer.biases_momentums = biases_updates
        else:
            weights_updates = -self._current_learning_rate * layer.d_weights
            biases_updates = self._current_learning_rate * layer.d_biases

        layer.weights += weights_updates
        layer.biases += biases_updates

    def update_learning_rate(self):
        if self._decay:
            self._current_learning_rate = self._learning_rate * (1. / (1. + self._decay * self._iteration))

        self._iteration += 1


class AdaGrad:

    def __init__(self, learning_rate=1., decay=0, momentum=0):
        if learning_rate < 0.:
            raise ValueError('Learning rate cant be less than 0')
        if decay < 0.:
            raise ValueError('Decay cant be less than 0')
        if momentum < 0.:
            raise ValueError('Momentum cant be less than 0')

        self._learning_rate = float(learning_rate)
        self._current_learning_rate = float(learning_rate)
        self._decay = decay
        self._iteration = 0
        self._momentum = momentum

    def update_params(self, layer):
        # Momentum is parameters change from previous iteration * self._momentum
        if self._momentum:
            weights_updates = self._momentum * layer.weights_momentums - self._current_learning_rate * layer.d_weights
            layer.weight_momentums = weights_updates
            biases_updates = self._momentum * layer.biases_momentums - self._current_learning_rate * layer.d_biases
            layer.biases_momentums = biases_updates
        else:
            weights_updates = -self._current_learning_rate * layer.d_weights
            biases_updates = self._current_learning_rate * layer.d_biases

        layer.weights += weights_updates
        layer.biases += biases_updates

    def update_learning_rate(self):
        if self._decay:
            self._current_learning_rate = self._learning_rate * (1. / (1. + self._decay * self._iteration))

        self._iteration += 1
