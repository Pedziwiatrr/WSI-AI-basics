import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def error_plot(y_test, predictions):
    matplotlib.use('TkAgg')
    true_qualities = np.array(y_test, dtype=float)
    predicted_qualities = np.array(predictions, dtype=float)

    unique_qualities = np.unique(true_qualities)
    avg_error_per_quality = []

    sample_counts = []
    for quality in unique_qualities:
        errors = [abs(prediction - quality) for true_quality, prediction in zip(true_qualities, predicted_qualities) if true_quality == quality]
        count = len(errors)
        sample_counts.append(count)
        if len(errors) > 0:
            avg_error = np.mean(errors)
        else:
            avg_error = 0
        avg_error_per_quality.append(avg_error)

    plt.figure(figsize=(8, 8))
    bars = plt.bar(unique_qualities, avg_error_per_quality, color='red')

    for bar in bars:
        error_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, error_val + 0.02,
                 round(error_val, 4), ha='center', va='bottom', fontsize=10)

    count_labels = [f'{quality} ({count})' for quality, count in zip(unique_qualities, sample_counts)]
    plt.xlabel('True Quality (sample count)', fontsize=14, labelpad=20)
    plt.ylabel('Average error', fontsize=14)
    plt.title('Average prediction error per quality', fontsize=18)
    plt.xticks(unique_qualities, count_labels,)
    plt.grid(axis='y')
    plt.show()

def compare_loss_plot(results):
    matplotlib.use('TkAgg')
    epochs_list = [result['epochs'] for result in results]
    test_losses = [result['test_loss'] for result in results]
    train_losses = [result['train_loss'] for result in results]
    plt.ylim(0, min(max(test_losses), max(train_losses), 5))

    plt.plot(epochs_list, test_losses, label='Test Loss', color='r')
    plt.plot(epochs_list, train_losses, label='Train Loss',  color='b')

    plt.xlabel('Epochs', fontsize=14, labelpad=20)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss in train/test samples', fontsize=18)
    plt.legend()
    plt.show()



