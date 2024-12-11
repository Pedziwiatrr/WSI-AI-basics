import numpy as np


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

def evaluate_total_cost(X, y, weights):
    """ calculate total cost of all values in data """
    total_cost = 0
    for i in range(len(y)):
        predicted_value = predict_value(X.iloc[i], weights)
        # if the prediction was wrong, raise cost value (logx < 0 for x < 1 )
        if y.iloc[i] == 1:
            total_cost -= np.log(predicted_value)
        else:
            total_cost -= np.log(1 - predicted_value)
    total_cost /= len(y)
    #print(total_cost)
    return total_cost

def adjust_weights(X, y, weights, step_length=0.01):
    """ adjust weights values using gradient descent algorithm to minimize cost """
    # gradient multipliers for every features weight
    weights_gradients = [0] * len(weights)
    for i in range(len(y)):
        predicted_value = predict_value(X.iloc[i], weights)
        error = predicted_value - y.iloc[i]
        for j in range(len(weights)):
            # error serves a role of a derivative. Bigger error -> needs a bigger value change
            weights_gradients[j] += error * X.iloc[i, j]
    for i in range(len(weights)):
        # Wi = Wi - 1/n * sum(predicted_value(xi) - y)xi
        weights[i] -= step_length * weights_gradients[i] / len(y)
    return weights

def learn(data, iterations, step_length=None):
    """ adjust weights by minimizing the cost (diagnosis error) """
    [X_train, X_test, y_train, y_test] = data
    weights = []
    for feature in X_train:
        weights.append(1)
    print (f"len(weights): {len(weights)}")
    for i in range(iterations):
        weights = adjust_weights(X_train, y_train, weights, step_length)
        if i % 100 == 0:
            print(f"Iteration: {i}, Current cost: {evaluate_total_cost(X_train, y_train, weights)}")
            print(f"Weights: {weights}\n")

    final_cost = evaluate_total_cost(X_train, y_train, weights)
    print(f"Final learning cost: {final_cost}")
    return weights

def predict(data, weights):
    """ use weights from learn function to diagnose cases from the rest of the data"""
    [X_train, X_test, y_train, y_test] = data
    diagnosis_list = []
    for id, case in X_test.iterrows():
        #print(f"Current case: {case}")
        prediction = predict_value(case, weights)
        diagnosis = round_to_binary(prediction)
        diagnosis_list.append(diagnosis)
    print(f"Diagnosis_list: {diagnosis_list}")
    return diagnosis_list

def logistic_regression(data, iterations, step_length=None):
    weights = learn(data, iterations, step_length)
    diagnosis_list = predict(data, weights)
    return diagnosis_list

