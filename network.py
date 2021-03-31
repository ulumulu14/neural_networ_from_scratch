class NeuralNetwork:

    def __init__(self):
        self._loss_func = None
        self._optimizer = None
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def set(self, loss, optimizier):
        self._loss_func = loss
        self._optimizer = optimizier

    def fit(self, X, y, epochs=1, validation_data=None):
        for epoch in range(1, epochs+1):
            regularization_loss = 0
            x = X

            # Forward pass
            for layer in self._layers:
                x = layer.forward(x)

            output = x

            # Regularization loss
            for layer in self._layers:
                if layer.trainable:
                    regularization_loss += self._loss_func.regularization_loss(layer)

            # Loss
            data_loss = self._loss_func.calculate(output, y)
            loss = data_loss + regularization_loss

            # Backward pass
            grad = self._loss_func.backward(output, y)

            for layer in reversed(self._layers):
                grad = layer.backward(grad)

            # Update parameters
            for layer in self._layers:
                if layer.trainable:
                    self._optimizer.update_params(layer)

            self._optimizer.update_learning_rate()

            val_loss = 0

            if validation_data is not None:
                X_val, y_val = validation_data
                x_val = X_val

                # Froward pass
                for layer in self._layers:
                    x_val = layer.forward(x_val, training=False)

                val_loss = self._loss_func.calculate(x, y_val)
            #print(loss)
            if epoch % 100 == 0:
                print(f'epoch: {epoch}/{epochs} ||  loss: {loss:.3f} || val_loss: {val_loss:.3f}')

    def structure(self):
        for i, layer in enumerate(self._layers):
            print(f"Layer {i + 1}")
            print(layer.get_details())