import layers


class NeuralNetwork():

    def __init__(self):
        self.data = None
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, learning_rate, iterations):
        self.data = X
        inputs = self.data

        ###for i in range(iterations):

        for layer in self.layers:
            inputs = layer.forward(inputs)

        outputs = inputs


            #reverse_inputs = outputs

            #for layer in reversed(self.layers):
             #   reverse_inputs = layer.backward(y, reverse_inputs, learning_rate)

            #for layer in self.layers:
             #   layer.read_input(inputs)
              #  inputs = layer.forward()

            #outputs = inputs

        return outputs

    def structure(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}")
            print(layer.get_details())

    def gradient_descent(self):
        pass

    def stochastic_gradient_descent(self):
        pass

    def adam(self):
        pass

