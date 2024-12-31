import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
from mlp import MLP


def compare(y_test, predictions, print_all=False):
    true_quality = np.array(y_test, dtype=float)
    predicted_quality = np.array(predictions, dtype=float)
    total_error = 0
    for true_quality, predicted_quality in zip(true_quality, predicted_quality):
        error = abs(true_quality - predicted_quality)
        if print_all:
            print(f"True Quality: {true_quality} // Predicted Quality: {predicted_quality} // Difference: {error}")
        total_error += error
    average_error = float(total_error / y_test.shape[0])
    return average_error


def test_params(X_train, X_test, y_train, y_test,
                epochs_list=[10, 100, 1000],
                learning_rates=[0.0001, 0.00001, 0.000001],
                hidden_layers_list=[[32], [32, 16], [64, 32, 16], [256, 128, 64, 32]]
                ):
    results = []
    param_combinations = itertools.product(epochs_list, learning_rates, hidden_layers_list)
    combination_count = len(epochs_list) * len(learning_rates) * len(hidden_layers_list)
    print(f"== Testing {combination_count} combinations ==\n")
    i = 0
    for epochs, learning_rate, hidden_layers in param_combinations:
        i += 1
        print(f"[{i}/{combination_count}] Currently testing: epochs: {epochs}, learning_rate: {learning_rate}, hidden_layers: {hidden_layers}")

        input_size = X_train.shape[1]
        output_size = 1
        mlp = MLP(input_size, hidden_layers, output_size, learning_rate)

        mlp.train(X_train, y_train, epochs)

        quality_predictions = mlp.predict(X_test)
        loss = mean_squared_error(y_test, quality_predictions)
        average_error = compare(y_test, quality_predictions, print_all=False)
        results.append({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "hidden_layers": hidden_layers,
            "loss": loss,
            "average_error": average_error
        })

    return results

def print_results(results):
    print("\n" + "="*75)
    print("== TEST RESULTS ==")
    print(f"{'Epochs':<10} {'Learning Rate':<15} {'Hidden Layers':<30} {'Loss':<10} {'Average Error':<10}")
    print(f"{'-' * 50}")
    for result in results:
        print(f"{result['epochs']:<10}"
              f"{result['learning_rate']:<15} "
              f"{str(result['hidden_layers']):<30}"
              f"{result['loss']:<10.4f} "
              f"{result['average_error']:<10.4f}")
    print("=" * 75)


