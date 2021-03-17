class SGD:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.d_weights
        layer.biases -= self.learning_rate * layer.d_biases
