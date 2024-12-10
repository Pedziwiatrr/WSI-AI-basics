import numpy as np


def sigmoid(value):
    # converts any value to probability (in range 0 to 1)
    return 1 / ( 1 + np.exp(-value))


def predict_value(X, weights):
    weighted_sum = 0
    for i in range(len(weights)):
        # predict y value based on all features (X) values and their weights
        weighted_sum += X[i] * weights[i]
    # convert predicted value to probability
    predicted_value = sigmoid(weighted_sum)
    return predicted_value


def evaluate_total_cost(X, y, weights):
    """ calculate total cost of all values in data """
    total_cost = 0
    for i in range(len(y)):
        predicted_value = predict_value(X[i], weights)
        # if the prediction was wrong, raise cost value (logx < 0 for x < 1 )
        if y[i] == 1:
            total_cost -= np.log(predicted_value)
        else:
            total_cost -= np.log(1 - predicted_value)
    total_cost /= len(y)
    return total_cost


def adjust_weights(X, y, weights, step_length):
    """ adjust weights values using gradient descent algorithm to minimize cost """
    # gradient multipliers for every features weight
    weights_gradients = [0] * len(weights)
    for i in range(len(y)):
        predicted_value = predict_value(X[i], weights)
        error = predicted_value - y[i]
        for j in range(len(weights)):
            # error serves a role of a derivative. Bigger error -> needs a bigger value change
            weights_gradients[j] += error * X[i][j]
    for i in range(len(weights)):
        # Wi = Wi - 1/n * sum(predicted_value(xi) - y)xi
        weights[i] -= step_length * weights_gradients[i] / len(y)
    return weights




