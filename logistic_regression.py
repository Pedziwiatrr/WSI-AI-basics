import numpy as np


def sigmoid(value):
    # converts any value to probability (in range 0 to 1)
    return 1 / ( 1 + np.exp(-value))


def calculate_cost(X, y, weights):
    total_cost = 0
    for i in range(len(y)):
        weighted_sum = 0
        feature_row = X[i]
        for j in range(len(weights)):
            weighted_sum += feature_row[j] * weights[j]
        predict_value = sigmoid(weighted_sum)
        if y[i] == 1:
            total_cost += -np.log(predict_value)
        else:
            total_cost += -np.log(1 - predict_value)
    total_cost /= len(y)
    return total_cost

def update_weights():
    pass

def train():
    pass

def predict_values():
    pass



