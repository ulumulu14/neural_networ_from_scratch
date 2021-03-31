import numpy as np


class Accuracy:

    def calculate(self, y_true, y_pred):
        return np.mean(self.compare(y_true, y_pred))

class RegressionAccuracy(Accuracy):
    # Not real accuracy for regression
    # Soft accuracy rounding values
    
    def compare(self, y_true, y_pred):
        return np.round(y_pred) == np.round(y_true)


class CategoricalAccuracy(Accuracy):

    def compare(self, y_true, y_pred):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        return y_pred == y_true


class BinaryAccuracy(Accuracy):

    def compare(self, y_true, y_pred):
        return y_pred == y_true