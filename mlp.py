import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_layers, output_size=1, learning_rate=0.01):
        self.input_size = input_size            # input_size = how many features in dataset
        self.hidden_layers = hidden_layers      # list which contains neurons amount in each hidden layer
        # for example [2, 1, 3] means that we want to create 3 hidden layers:
        # first layer - contains 2 neurons
        # second layer - contains 1 neuron
        # third layer - contains 3 neurons
        self.output_size = output_size          # output_size = how many neurons in output layer
                                                # we are predicting one value, so 1 is good choice
        self.learning_rate = learning_rate  # another parameter to adjust

        # weights and biases init
        self.weights = []
        self.biases = []

        # nie chce mi sie po angielsku juz pisac, nie mam c2 nawet
        # dla kazdej warstwy ukrytej robimy sobie macierz wag
        layer_input_size = input_size           # ilosc neuronow w poprzedniej warstwie

        for hidden_size in hidden_layers:
            # wagi na poczatku inicjujemy losowymi wartosciami od -1 do 1
            # macierz wag jest o rozmiarze (ilosc neuronow w poprzedniej warstwie) x (ilosc neuronow w aktualnej warstwie)
            self.weights.append(np.random.uniform(-0.1, 0.1, (layer_input_size, hidden_size)))
            # biasy sa po prostu inicjowane jako wektor zawierajacy tyle zer ile jest neuronow w danej warstwie
            self.biases.append(np.zeros((1, hidden_size)))
            layer_input_size = hidden_size
        # inicjalizacja wartwy wyjsciowej
        self.weights.append(np.random.uniform(-0.1, 0.1, (layer_input_size, output_size)))
        self.biases.append(np.zeros((1, output_size)))

        # dodalem to zeby warninga nie bylo, ale tak naprawde kazdy forward inicjuje wartosci od nowa
        self.outputs = []
        self.inputs = []
        self.output_input = []
        self.output = []

    def forward(self, X):
        self.outputs = []       # tu beda wyniki dla kazdej warstwy
        self.inputs = []        # tu beda dane wejsciowe dla każdej warstwy (czyli sumy wazone + bias)
        input_data = X
        for i in range(len(self.hidden_layers)):
            self.inputs.append(np.dot(input_data, self.weights[i]) + self.biases[i])
            output = sigmoid(self.inputs[-1])
            self.outputs.append(output)
            input_data = output
        # warstwa wyjsciowa
        self.output_input = np.dot(input_data, self.weights[-1]) + self.biases[-1]
        self.output = self.output_input  # brak aktywacji dla regresji
        return self.output

    def backward(self, X, y):
        # calculating the difference between prediction and y_train target values
        error = self.output - y
        error_gradients = error

        # ensuring output gradient is numpy array (it raises errors without it)
        if isinstance(error_gradients, pd.DataFrame):
            error_gradients = error_gradients.to_numpy()
        elif isinstance(error_gradients, pd.Series):
            error_gradients = error_gradients.to_numpy()

        # calculating the gradients for final (output) layers weights and biases

            # transposing the last hidden layer outputs so that its column count (batch size)
            # is equal to error_gradients row count
        weights_gradients = np.dot(self.outputs[-1].T, error_gradients)

            # bias gradients are just summed errors for every neuron in the current sample
        bias_gradients = np.sum(error_gradients, keepdims=True)

        # lists containing final (output) layers gradients, will store them for every other layer as well soon
        error_gradients_list = [error_gradients]
        weights_gradients_list = [weights_gradients]
        bias_gradients_list = [bias_gradients]

        # backpropagation through hidden layers
        for i in reversed(range(len(self.hidden_layers))):
            # calculating gradient for current hidden layer
            hidden_layer_gradients = np.dot(error_gradients_list[-1], self.weights[i + 1].T) * sigmoid_derivative(self.outputs[i])

            # calculating weight and bias gradients for current hidden layer
            if i > 0:
                # if we are in hidden layer use transposed output of the layer before the current one
                layer_weights_gradients = np.dot(self.outputs[i - 1].T, hidden_layer_gradients)
            else:
                # if we are in input layer use transposed X input instead
                layer_weights_gradients = np.dot(X.T, hidden_layer_gradients)
            layer_bias_gradients = np.sum(hidden_layer_gradients, keepdims=True)

            # updating gradients lists with current layers calculated values
            error_gradients_list.append(hidden_layer_gradients)
            weights_gradients_list.append(layer_weights_gradients)
            bias_gradients_list.append(layer_bias_gradients)

        # Updating biases and weights using previously calculated gradients
        for i in range(len(self.weights)):
            self.biases[i] -= self.learning_rate * bias_gradients_list[ -(i + 1) ]
            self.weights[i] -= self.learning_rate * weights_gradients_list[ -(i + 1) ]

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            if (epoch + 1) % 1 == 0:
                loss = mean_squared_error(y, output)
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)
