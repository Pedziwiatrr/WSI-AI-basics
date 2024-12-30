import argparse
from sklearn.metrics import mean_squared_error

from load_data import get_data, prepare_data
from mlp import MLP
from result_comparator import compare, compare_plot


def main():
    # arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--test_ratio', type=float, default=0.25)
    parser.add_argument('--hidden_layers', type=int, nargs='*', default=[2, 1, 3])
    args = parser.parse_args()

    # fetching data
    X, y = get_data()
    data = prepare_data(X, y, test_ratio=args.test_ratio, seed=args.seed)
    X_train, X_test, y_train, y_test = data

    # initializing multilayer perceptron
    input_size = X_train.shape[1]           # features count in X
    hidden_layers = args.hidden_layers
    output_size = 1                         # target count in y
    learning_rate = args.learning_rate
    mlp = MLP(input_size, hidden_layers, output_size, learning_rate)

    print("\n== Initialization parameters ==")
    print(f"Input size: {input_size}\nOutput size: {output_size}")
    print(f"Learning rate: {learning_rate}\nHidden layers: {hidden_layers}\n")

    # training mlp (learning phase)
    mlp.train(X_train, y_train, epochs=args.epochs)

    # testing mlp (working phase)
    quality_predictions = mlp.predict(X_test)
    loss = mean_squared_error(y_test, quality_predictions)
    print(f"Final loss: {loss}")
    compare(y_test, quality_predictions, print_all=False)
    compare_plot(y_test, quality_predictions)


if __name__ == '__main__':
    main()
