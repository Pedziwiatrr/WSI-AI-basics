import matplotlib.pyplot as plt
import numpy as np


def compare(y_test, predictions, print_all=False):
    true_quality = np.array(y_test, dtype=float)
    predicted_quality = np.array(predictions, dtype=float)
    total_error = 0
    for true_quality, predicted_quality in zip(true_quality, predicted_quality):
        error = abs(true_quality - predicted_quality)
        if print_all:
            print(f"True Quality: {true_quality} // Predicted Quality: {predicted_quality} // Difference: {error}")
        total_error += error
    print(f"Average prediction error: {total_error / y_test.shape[0]}")