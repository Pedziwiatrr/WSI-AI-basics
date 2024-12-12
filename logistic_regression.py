import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def round_to_binary(value):
    if value >= 0.5:
        return 1
    else:
        return 0

def sigmoid(value):
    # converts any value to probability (in range 0 to 1)
    return 1 / ( 1 + np.exp(-value))

def predict_value(X, weights):
    weighted_sum = 0
    for i in range(len(weights)):
        # predict y value based on all features (X) values and their weights
        weighted_sum += X.iloc[i] * weights[i]
    # convert predicted value to probability
    predicted_value = sigmoid(weighted_sum)

    if predicted_value == 0:
        predicted_value = 1e-6
    elif predicted_value == 1:
        predicted_value = 1 - 1e-6
    #print(predicted_value)
    return predicted_value

def adjust_weights(X, y, weights, step_length=0.001):
    """ adjust weights values using gradient descent algorithm to minimize cost """
    # gradient multipliers for every features weight
    weights_gradients = [0] * len(weights)
    total_error = 0
    for i in range(len(y)):
        predicted_value = predict_value(X.iloc[i], weights)
        error = predicted_value - y.iloc[i]
        total_error += error
        for j in range(len(weights)):
            # error serves a role of a derivative. Bigger error -> needs a bigger value change
            weights_gradients[j] += error * X.iloc[i, j]
    for i in range(len(weights)):
        # Wi = Wi - 1/n * sum(predicted_value(xi) - y)xi
        weights[i] -= step_length * weights_gradients[i] / len(y)
    return weights, total_error

def learn(data, iterations, step_length, normalize=False):
    """ adjust weights by minimizing the cost (diagnosis error) """
    [X_train, X_test, y_train, y_test] = data
    weights = []
    for feature in X_train:
        weights.append(0)
    #print (f"len(weights): {len(weights)}")
    for i in range(iterations):
        weights, error = adjust_weights(X_train, y_train, weights, step_length)
        if i % 50 == 0:
            print(f"Iteration: {i}, Error: {error:.2f}")
            #print(f"Weights: {weights}\n")
    return weights

def predict(data, weights):
    """ use weights from learn function to diagnose cases from the rest of the data"""
    [X_train, X_test, y_train, y_test] = data
    diagnosis_list = []
    predicted_probabilities = []
    for id, case in X_test.iterrows():
        #print(f"Current case: {case}")
        prediction = predict_value(case, weights)
        predicted_probabilities.append(prediction)
        diagnosis = round_to_binary(prediction)
        diagnosis_list.append(diagnosis)
    #print(f"Diagnosis_list: {diagnosis_list}")
    return diagnosis_list, predicted_probabilities

def logistic_regression(data, iterations, step_length, normalize=False):
    weights = learn(data, iterations, step_length, normalize)
    diagnosis_list, predicted_probabilities = predict(data, weights)
    return diagnosis_list, predicted_probabilities

