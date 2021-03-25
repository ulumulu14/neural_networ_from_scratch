import network
import layers
import activations
import losses
import optimizers
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from sklearn import datasets

nnfs.init()


if __name__ == "__main__":
    EPOCHS = 10001
    LEARNING_RATE = 0.02

    #X = np.array([[1, 2, 3, 2.5],
    #              [2.0, 5.0, -1.0, 2.0],
    #              [-1.5, 2.7, 3.3, -0.8]])
    #X1 = np.array([1, 2, 3, 2.5])
    #y = np.array([[1, 0, 0],
    #              [0, 1, 0],
    #              [0, 0, 1]])

    X, y = spiral_data(samples=1000, classes=3)

    #iris = datasets.load_iris()
    #X = iris.data[:, :2]
    #y = iris.target

    dense1 = layers.Dense(2, 512, weight_regularizer_l2=0.0005, bias_regularizer_l2=0.0005)
    relu1 = activations.ReLU()
    dropout1 = layers.Dropout(0.1)
    dense2 = layers.Dense(512, 3)
    #s1 = activations.Softmax()
    loss_activation = losses.Softmax_CategoricalCrossentropy()
    #loss_function = losses.CategoricalCrossentropy()
    #optimizer = optimizers.SGD(learning_rate=LEARNING_RATE, decay=0.001, momentum=0.9)
    #optimizer = optimizers.AdaGrad(learning_rate=LEARNING_RATE, decay=0.0001)
    optimizer = optimizers.RMSProp(learning_rate=LEARNING_RATE, decay=0.0001, rho=0.999)
    #optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, decay=0.00001)

    for epoch in range(EPOCHS):
        x = dense1.forward(X)
        x = relu1.forward(x)
        x = dropout1.forward(x)
        x = dense2.forward(x)
        #output = s1.forward(x)
        loss = loss_activation.forward(x, y)
        #loss = loss_function.calculate(output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        #predictions = np.argmax(output, axis=1)

        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == y)

        if epoch % 100 == 0:
            print(f'epoch: {epoch}/{EPOCHS} || accuracy: {accuracy:.3f} || loss: {loss:.3f}')

        #grad = loss_function.backward(output, y)
        grad = loss_activation.backward(loss_activation.output, y)
        #grad = s1.backward(grad)

        grad = dense2.backward(grad)
        grad = dropout1.backward(grad)
        grad = relu1.backward(grad)
        grad = dense1.backward(grad)


        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_learning_rate()

    X_test, y_test = spiral_data(samples=100, classes=3)

    x = dense1.forward(X_test)
    x = relu1.forward(x)
    x = dense2.forward(x)
    val_loss = loss_activation.forward(x, y_test)

    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)

    val_acc = np.mean(predictions == y_test)

    print(f'val_acc: {val_acc} || val_loss: {val_loss}')

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





