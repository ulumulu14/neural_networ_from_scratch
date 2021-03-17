import network
import layers
import activations
import losses
import optimizers
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


if __name__ == "__main__":

    X = np.array([[1, 2, 3, 2.5],
                  [2.0, 5.0, -1.0, 2.0],
                  [-1.5, 2.7, 3.3, -0.8]])
    X1 = np.array([1, 2, 3, 2.5])
    y = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    X, y = spiral_data(samples=100, classes=3)

    d1 = layers.Dense(2, 64)
    r1 = activations.ReLU()
    d2 = layers.Dense(64, 128)
    r2 = activations.ReLU()
    d3 = layers.Dense(128, 64)
    r3 = activations.ReLU()
    d4 = layers.Dense(64, 3)
    s1 = activations.Softmax()

    loss_function = losses.CategoricalCrossentropy()
    optimizer = optimizers.SGD(learning_rate=0.1)

    for epoch in range(10001):
        x = d1.forward(X)
        x = r1.forward(x)
        x = d2.forward(x)
        x = r2.forward(x)
        x = d3.forward(x)
        x = r3.forward(x)
        x = d4.forward(x)
        output = s1.forward(x)

        loss = loss_function.calculate(output, y)

        predictions = np.argmax(output, axis=1)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        accuracy = np.mean(predictions==y)

        if epoch % 100 == 0:
            print(f'epoch: {epoch} || accuracy: {accuracy:.3f} || loss: {loss:.3f}')

        grad = loss_function.backward(output, y)
        grad = s1.backward(grad)
        grad = d4.backward(grad)
        grad = r3.backward(grad)
        grad = d3.backward(grad)
        grad = r2.backward(grad)
        grad = d2.backward(grad)
        grad = r1.backward(grad)
        grad = d1.backward(grad)

        optimizer.update_params(d1)
        optimizer.update_params(d2)
        optimizer.update_params(d3)
        optimizer.update_params(d4)



'''''
    nn = network.NeuralNetwork()

    nn.add_layer(layers.Dense(5))
    nn.add_layer(activations.ReLU())
    nn.add_layer(layers.Dense(3))
    nn.add_layer(activations.Softmax())

    y_pred = nn.fit(X, None, None, None)

    loss_function = losses.CategoricalCrossentropy()
    #loss = loss_function.calculate(y_pred, y)

    print(y_pred)
    print(nn.structure())
    print(f'loss: {loss}')
    #nn.add_layer(5, "relu")
    #nn.add_layer(2, "linear")
    #nn.add_layer(5, "relu")
    #nn.add_layer(2, "softmax")

    #nn.structure()

    #output = nn.train(X)




    #print(output)

    # Test
    softmax_output = np.array([[0.7, 0.1, 0.2],
                               [0.1, 0.5, 0.4],
                               [0.02, 0.9, 0.08]])
    class_targets = np.array([0, 1, 1])

    activation = activations.Softmax()
    activation.output = softmax_output
    loss = losses.CategoricalCrossentropy()
    loss_grad = loss.backward(softmax_output, class_targets)
    softmax_grad = activation.backward(loss_grad)
    print(softmax_grad)
    '''''





