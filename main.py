import network
import layers
import activations
import losses
import numpy as np

if __name__ == "__main__":

    X = np.array([[1, 2, 3, 2.5],
                  [2.0, 5.0, -1.0, 2.0],
                  [-1.5, 2.7, 3.3, -0.8]])
    X1 = np.array([1, 2, 3, 2.5])
    y = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    



    nn = network.NeuralNetwork()

    nn.add_layer(layers.Dense(5))
    nn.add_layer(activations.ReLU())
    nn.add_layer(layers.Dense(3))
    nn.add_layer(activations.Softmax())

    y_pred = nn.fit(X, None, None, None)

    loss_function = losses.CategoricalCrossentropy()
    loss = loss_function.calculate(y_pred, y)

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
    '''''
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
