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
    '''
    print(X)
    layer1 = DenseLayer(4, "linear")
    layer1.read_input(X)
    output1 = layer1.forward()
    print(output1)

    layer2 = DenseLayer(4, "relu")
    layer2.read_input(output1)
    output2 = layer1.forward()
    print(output2)
    '''

    nn = network.NeuralNetwork()

    nn.add_layer(layers.Dense(5))
    nn.add_layer(activations.ReLU())
    nn.add_layer(layers.Dense(3))
    nn.add_layer(activations.Softmax())

    y_pred = nn.fit(X, None, None, None)

    loss_function = losses.CategoricalCrossEntropy()
    loss = loss_function.calculate(y_pred, y)
    print(f'loss: {loss}')
    print(y_pred)
    print(nn.structure())
    #nn.add_layer(5, "relu")
    #nn.add_layer(2, "linear")
    #nn.add_layer(5, "relu")
    #nn.add_layer(2, "softmax")

    #nn.structure()

    #output = nn.train(X)




    #print(output)
