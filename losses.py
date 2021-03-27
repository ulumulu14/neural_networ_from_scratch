import numpy as np
import activations
import layers


class Loss:

    def calculate(self, y_pred, y_true):
        return np.mean(self.forward(y_pred, y_true))

    def regularization_loss(self, layer):
        # L1 and L2 regularizations are penalty for big weights and biases added to loss
        # It make model generalize better
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights*layer.weights)

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases*layer.biases)

        return regularization_loss


class CategoricalCrossentropy(Loss):

    def __init__(self):
        self._d_inputs = None

    def forward(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        n_samples = len(y_pred)

        # Clip to prevent division by 0
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # For classes as numbers
        if len(y_true.shape) == 1:
            confidences = y_pred[range(n_samples), y_true]
        # For one-hot encoded classes
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred * y_true)
        else:
            raise Exception('Wrong y shape')

        return -np.log(confidences)

    def backward(self, y_pred, y_true):
        samples = len(y_pred)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self._d_inputs = y_pred.copy()
        self._d_inputs[range(samples), y_true] -= 1
        self._d_inputs = self._d_inputs / samples

        return self._d_inputs

    '''''
    def backward(self, y_pred, y_true):
        
        n_samples = len(y_pred)
        labels = len(y_pred[0])
        #y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        

        # One-hot encoding
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Return normalized gradient
        return (-y_true / y_pred)/n_samples
        '''''

class Softmax_CategoricalCrossentropy:
    #Softmax and CategoricalCrossentropy combined for optimization purposes
    def __init__(self):
        self.activation = activations.Softmax()
        self.loss = CategoricalCrossentropy()
        self._d_inputs = None

    def forward(self, inputs, y_true):
        self.output = self.activation.forward(inputs)

        return self.loss.calculate(self.output, y_true)

    def backward(self, d_values, y_true):
        samples = len(d_values)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self._d_inputs = d_values.copy()
        self._d_inputs[range(samples), y_true] -= 1
        self._d_inputs = self._d_inputs / samples

        return self._d_inputs
