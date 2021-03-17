import numpy as np
import layers
#from abc import abstractmethod


class Loss:

    def calculate(self, y_pred, y_true):
        return np.mean(self.forward(y_pred, y_true))


class CategoricalCrossentropy(Loss):

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
        n_samples = len(y_pred)
        labels = len(y_pred[0])

        # One-hot encoding
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Retrun normilzed gradient
        return (-y_true / y_pred)/n_samples

