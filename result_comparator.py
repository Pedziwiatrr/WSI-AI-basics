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
    print(f"\n== Average prediction error: {total_error / y_test.shape[0]} ==")


def compare_plot(y_test, predictions):
    true_qualities = np.array(y_test, dtype=float)
    predicted_qualities = np.array(predictions, dtype=float)

    # getting every quality once for x axis
    unique_qualities = np.unique(true_qualities)
    avg_error_per_quality = []

    for quality in unique_qualities:
        errors = [abs(prediction - quality) for true_quality, prediction in zip(true_qualities, predicted_qualities) if true_quality == quality]
        if len(errors) > 0:
            avg_error = np.mean(errors)
        else:
            avg_error = 0
        avg_error_per_quality.append(avg_error)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_qualities, avg_error_per_quality, color='red')
    plt.xlabel('True Quality')
    plt.ylabel('Average error')
    plt.title('Average prediction error per quality')
    plt.xticks(unique_qualities)
    plt.grid(axis='y')
    plt.show()