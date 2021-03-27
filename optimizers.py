import numpy as np


class SGD:

    def __init__(self, learning_rate=0.001, decay=0, momentum=0):
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
        # Used to prevent stopping in loss function's local minimum
        if self._momentum:
            if layer.weights_momentums is None:
                layer.weights_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)

            weights_updates = self._momentum*layer.weights_momentums - self._current_learning_rate*layer.d_weights
            layer.weight_momentums = weights_updates
            biases_updates = self._momentum*layer.biases_momentums - self._current_learning_rate*layer.d_biases
            layer.biases_momentums = biases_updates
        else:
            weights_updates = -self._current_learning_rate * layer.d_weights
            biases_updates = -self._current_learning_rate * layer.d_biases

        layer.weights += weights_updates
        layer.biases += biases_updates

    def update_learning_rate(self):
        # Call after updating parameters of all layers
        if self._decay:
            self._current_learning_rate = self._learning_rate * (1. / (1. + self._decay*self._iteration))

        self._iteration += 1


class AdaGrad:

    def __init__(self, learning_rate=0.001, decay=0, epsilon=0.0000001):
        if learning_rate < 0.:
            raise ValueError('Learning rate cant be less than 0')
        if decay < 0.:
            raise ValueError('Decay cant be less than 0')
        if epsilon <= 0.:
            raise ValueError('Epsilon cant be lower or equal to 0')

        self._learning_rate = float(learning_rate)
        self._current_learning_rate = float(learning_rate)
        self._decay = float(decay)
        self._iteration = 0
        self._epsilon = float(epsilon)  # To prevent dividing by 0

    def update_params(self, layer):
        # Weights and biases change is lowered by previous gradients,
        # this normalizes changes (per weight adaptive learning rate)
        if layer.d_weights_history is None:
            layer.d_weights_history = np.zeros_like(layer.weights)
            layer.d_biases_history = np.zeros_like(layer.biases)

        layer.d_weights_history += layer.d_weights**2
        layer.d_biases_history += layer.d_biases**2

        layer.weights -= self._current_learning_rate * layer.d_weights / (np.sqrt(layer.d_weights_history)+self._epsilon)
        layer.biases -= self._current_learning_rate * layer.d_biases / (np.sqrt(layer.d_biases_history)+self._epsilon)

    def update_learning_rate(self):
        # Call after updating parameters of all layers
        if self._decay:
            self._current_learning_rate = self._learning_rate * (1. / (1. + self._decay * self._iteration))

        self._iteration += 1


class RMSProp:

    def __init__(self, learning_rate=0.001, decay=0, epsilon=0.0000001, rho=0.9):
        if learning_rate < 0.:
            raise ValueError('Learning rate cant be less than 0')
        if decay < 0.:
            raise ValueError('Decay cant be less than 0')
        if epsilon <= 0.:
            raise ValueError('Epsilon cant be lower or equal to 0')
        if rho < 0.:
            raise ValueError('Rho cant be lower than 0')

        self._learning_rate = float(learning_rate)
        self._current_learning_rate = float(learning_rate)
        self._decay = float(decay)
        self._iteration = 0
        self._epsilon = float(epsilon)  # To prevent dividing by 0
        self._rho = float(rho)

    def update_params(self, layer):
        # Weights and biases change is lowered by previous gradients,
        # this normalizes changes (per weight adaptive learning rate)
        if layer.d_weights_history is None:
            layer.d_weights_history = np.zeros_like(layer.weights)
            layer.d_biases_history = np.zeros_like(layer.biases)

        layer.d_weights_history = self._rho * layer.d_weights_history + (1-self._rho) * layer.d_weights**2
        layer.d_biases_history = self._rho * layer.d_biases_history + (1-self._rho) * layer.d_biases**2

        layer.weights -= self._current_learning_rate * layer.d_weights / (np.sqrt(layer.d_weights_history)+self._epsilon)
        layer.biases -= self._current_learning_rate * layer.d_biases / (np.sqrt(layer.d_biases_history)+self._epsilon)

    def update_learning_rate(self):
        # Call after updating parameters of all layers
        if self._decay:
            self._current_learning_rate = self._learning_rate * (1. / (1. + self._decay * self._iteration))

        self._iteration += 1


class Adam:

    def __init__(self, learning_rate=0.001, decay=0, epsilon=0.0000001, beta1=0.9, beta2=0.999):
        if learning_rate < 0.:
            raise ValueError('Learning rate cant be less than 0')
        if decay < 0.:
            raise ValueError('Decay cant be less than 0')
        if epsilon <= 0.:
            raise ValueError('Epsilon cant be lower or equal to 0')
        if beta1 < 0.:
            raise ValueError('beta1 cant be lower than 0')
        if beta2 < 0.:
            raise ValueError('beta2 cant be lower than 0')

        self._learning_rate = float(learning_rate)
        self._current_learning_rate = float(learning_rate)
        self._decay = float(decay)
        self._iteration = 0
        self._epsilon = float(epsilon)  # To prevent dividing by 0
        self._beta1 = float(beta1)
        self._beta2 = float(beta2)

    def update_params(self, layer):
        # Weights and biases change is lowered by previous gradients,
        # this normalizes weight changes (per weight adaptive learning rate)
        # betas compensate initial zeroed values of momentums and history of weights and biases
        if layer.d_weights_history is None:
            layer.d_weights_history = np.zeros_like(layer.weights)
            layer.d_biases_history = np.zeros_like(layer.biases)
            layer.weights_momentums = np.zeros_like(layer.weights)
            layer.biases_momentums = np.zeros_like(layer.biases)

        layer.weights_momentums = self._beta1 * layer.weights_momentums + (1+self._beta1) * layer.d_weights
        layer.biases_momentums = self._beta1 * layer.biases_momentums + (1+self._beta1) * layer.d_biases
        print(layer.weights_momentums)
        weights_momentums_corrected = layer.weights_momentums / (1 - self._beta1**(self._iteration+1))
        biases_momentums_corrected = layer.biases_momentums / (1 - self._beta1**(self._iteration+1))

        layer.d_weights_history = self._beta2 * layer.d_weights_history + (1-self._beta2) * layer.d_weights**2
        layer.d_biases_history = self._beta2 * layer.d_biases_history + (1-self._beta2) * layer.d_biases**2

        layer.d_weights_history_corrected = layer.d_weights_history / (1 - self._beta2**(self._iteration+1))
        layer.d_biases_history_corrected = layer.d_biases_history / (1 - self._beta2**(self._iteration+1))

        layer.weights -= self._current_learning_rate * weights_momentums_corrected / \
                         (np.sqrt(weights_momentums_corrected)+self._epsilon)
        layer.biases -= self._current_learning_rate * biases_momentums_corrected / \
                        (np.sqrt(biases_momentums_corrected)+self._epsilon)

    def update_learning_rate(self):
        # Call after updating parameters of all layers
        if self._decay:
            self._current_learning_rate = self._learning_rate * (1. / (1. + self._decay * self._iteration))

        self._iteration += 1

