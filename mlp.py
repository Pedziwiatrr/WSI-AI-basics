import numpy as np
from sklearn.metrics import mean_squared_error


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

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
        self.learning_rate = learning_rate      # another parameter to adjust

        # weights and biases init
        self.weights = []
        self.biases = []

        layer_input_size = input_size           # amount of neurons in previous layer

        for hidden_size in hidden_layers:
            # weights are random values between -0.1 and 0.1
            # every layer has it's own array of weights
            # weights array shape: (amount of neurons in previous layer) x (amount of neurons in current layer)
            # that's because every column represents set of weights for each neuron
            # for example one of our hidden layers has following weights array:
            # [
            #   [0.15, 0.4, 0.8],
            #   [0.12, 0.3, 0.2],
            #   [0.88, 1.2, 1.3],
            #   [0.99, 0.5, 0.3]
            # ]
            # it means that first neuron's set of weight is [0.15, 0.12, 0.88, 0.99]
            # second one has [0.4, 0.3, 1.2, 0.5] and third one [0.8, 0.2, 1.3, 0.3]
            self.weights.append(np.random.uniform(-0.1, 0.1, (layer_input_size, hidden_size)))
            # every neuron in layer has one bias, so for every layer
            # biases are represented by vector of values, for example:
            # [0.12, 0.15, 0.9]
            # means that neuron1 has bias = 0.12, neuron2 has bias = 0.15 and neuron3 has bias = 0.9
            self.biases.append(np.zeros((1, hidden_size)))
            # every single bias is initialized as 0
            layer_input_size = hidden_size
        # initialization of output layer
        self.weights.append(np.random.uniform(-0.1, 0.1, (layer_input_size, output_size)))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, X):
        self.outputs = []  # list of outputs for each layer
        self.inputs = []  # input data for each layer (weighted_sum + bias)
        input_data = X  # given inputs

        for i in range(len(self.hidden_layers)):
            # for each neuron in layer we have to compute following equation:
            # weighted_sum = feature1 * weight1 + feature2 * weight2 + ... + featureN * weightN + bias
            # let's say that our neural network has 3 inputs
            # and in current hidden layer there are 4 neurons, which contains following data:
            # input_data = [
            #   [1, 2, 3],
            #   [4, 5, 6],
            #   [7, 8, 9],
            #   [2, 4, 6],
            #   [8, 9, 3]
            # ]
            # (5 samples with 3 inputs each)
            # weights[current_layer] = [
            #   [0.5, 0.12, 0.88, 0.99],
            #   [0.9, 1.33, 0.72, 0.08],
            #   [0.3, 2.11, 3.33, 0.18]
            # ]
            # biases[current_layer] = [9, 8, 7, 6]
            # then this code:
            weighted_sum = np.dot(input_data, self.weights[i]) + self.biases[i]
            # returns:
            # weighted_sum = [
            #   [1 * 0.5 + 2 * 0.9 + 3 * 0.3 + 9, 1 * 0.12 + 2 * 1.33 + 3 * 2.11 + 8, ... ],
            #   [4 * 0.5 + 5 * 0.9 + 6 * 0.3 + 9, 4 * 0.12 + 5 * 1.33 + 6 * 2.11 + 8, ... ],
            #   ...
            #   [2 * 0.5 + 4 * 0.9 + 6 * 0.3 + 9, 2 * 0.12 + 4 * 1.33 + 6 * 2.11 + 8, ... ],
            #   [8 * 0.5 + 9 * 0.9 + 3 * 0.3 + 9, 8 * 0.12 + 9 * 1.33 + 3 * 2.11 + 8, ... ]
            # ]
            # so weighted sum is output of current layer and simultaneously input for next layer
            self.inputs.append(weighted_sum)
            # before we use activation function, we have to save this value (for backpropagation)

            # Now we have to use activation function, which in our case is relu
            # relu is literally max(0, weighted_sum) function,
            # so it obviously replaces any negative value with 0
            output = relu(weighted_sum)
            self.outputs.append(output)
            # why are we doing that?
            # negative neuron output suggests that this neuron has
            # negative impact on learning process, so we can ignore it
            # moreover it makes our model non-linear
            # however we have to keep in mind that replacing output with zero
            # makes our neuron "dead" and it has no longer any impact on our neural network
            # too many dead neurons makes neural network useless
            input_data = output     # input update

        # we don't use activation function on output layer, because
        # we are computing regression, not classification
        self.output = np.dot(input_data, self.weights[-1]) + self.biases[-1]
        return self.output

    def backward(self, X, y):
        # converting data to numpy
        X = np.asarray(X)
        y = np.asarray(y)

        error = ( self.output - y ) / y.shape[0]
        print(error)
        error_gradients = error
        # self.output shape: (samples_count) x (output_size)
        # we set neuron_count in output layer to 1,
        # so shape of our output is (samples_count) x 1
        # y is just a vector of actual values
        # so we can simply subtract one vector from another

        weights_gradients = np.dot(self.outputs[-1].T, error_gradients)
        # self.outputs[-1] = [output1, output2, ..., outputN]
        # weight_gradients = [output1 * error_gradients1, output2 * error_gradients2, ..., outputN * error_gradientsN]
        bias_gradients = np.sum(error_gradients, axis=0, keepdims=True)
        # bias_gradients = [[ sum_of_error_gradients ]]

        # we will store gradients in lists
        weights_gradients_list = [weights_gradients]
        bias_gradients_list = [bias_gradients]
        error_gradients_list = [error_gradients]

        # backpropagation
        for i in reversed(range(len(self.hidden_layers))):
            # error gradients for the current hidden layer
            hidden_layer_gradients = (
                    np.dot(error_gradients_list[-1], self.weights[i + 1].T) * relu_derivative(self.inputs[i])
            )

            # current layers weights gradients
            if i > 0:
                layer_weights_gradients = np.dot(self.outputs[i - 1].T, hidden_layer_gradients)
            else:
                layer_weights_gradients = np.dot(X.T, hidden_layer_gradients)

            # current layers bias gradients
            layer_bias_gradients = np.sum(hidden_layer_gradients, axis=0, keepdims=True)

            # updating lists with gradients for all layers with current layer gradients values
            error_gradients_list.append(hidden_layer_gradients)
            weights_gradients_list.append(layer_weights_gradients)
            bias_gradients_list.append(layer_bias_gradients)

        # updating weights and biases in every layer
        for i in range(len(self.weights)):
            self.biases[i] -= self.learning_rate * bias_gradients_list[-(i + 1)]
            self.weights[i] -= self.learning_rate * weights_gradients_list[-(i + 1)]

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            if (epoch + 1) % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f"    Epoch {epoch + 1}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)
