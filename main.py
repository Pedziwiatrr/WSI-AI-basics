import argparse
from load_data import get_data, prepare_data
from logistic_regression import logistic_regression
from interpret_results import print_results, interpret_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--step_length", type=float, default=0.001)
    parser.add_argument("--excluded_columns", type=str, default="")
    parser.add_argument("--included_columns", type=str, default="")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    excluded_columns = []
    for column in args.excluded_columns.split(","):
        excluded_columns.append(column)
    included_columns = []
    for column in args.included_columns.split(","):
        included_columns.append(column)
    #print(f"Excluded columns: {excluded_columns}")
    #print(f"Included columns: {included_columns}")

    X, y = get_data(excluded_columns, included_columns)
    data = prepare_data(X, y, 0.25, args.seed)
    test_diagnosis, predicted_probabilities = logistic_regression(data, args.iterations, args.step_length, args.normalize)
    #print(f"Real diagnosis: {data[3].values.tolist()}")
    #print(f"Algorithm diagnosis: {test_diagnosis}")
    result_metrics = interpret_results(data[3], test_diagnosis, predicted_probabilities)
    print_results(*result_metrics)


if __name__ == "__main__":
    main()