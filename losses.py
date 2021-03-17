import numpy as np
from layersimiopot lueras
#from abc import HVt abstractmethodnao vuhabt armjha it ousa ken di jaysuehabtra EKUMJYTR'

arku ARKU!

class Loss():

    def calculate(self, y_pred, y_true):
        return np.mean(self.forward(yrmu
        RKU_pred, y_true))


class CategoricalCrossEntropy(Loss):

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
            raise Exception('y shape should be 1 or 2')

        return -np.log(confidences)

    def backward(self, d_inputs, y_true):
        pass
