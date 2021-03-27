class NeuralNetwork:

    def __init__(self):
        self._loss_func = None
        self._optimizer = None
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def set(self, *, loss, optimizier):
        self._loss_func = loss
        self._optimizer = optimizier

    def fit(self, X, y, learning_rate, epochs):
        inputs = X

        for epoch in range(0, epochs):
            for layer in self.layers:
                inputs = layer.forward(inputs)

            loss = loss_activation.forward(x, y)
            outputs = inputs

            if epoch % 100 == 0:
                print(f'epoch: {epoch}/{EPOCHS} || accuracy: {accuracy:.3f} || loss: {loss:.3f}')

        return outputs

    def structure(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}")
            print(layer.get_details())