import argparse
from load_data import get_data, prepare_data
from logistic_regression import logistic_regression


def compare_diagnoses(first_diagnosis, second_diagnosis):
    same_verdicts = 0
    different_verdicts = 0
    for i in range(len(first_diagnosis)):
        verdict1 = first_diagnosis.iloc[i]
        verdict2 = second_diagnosis[i]
        print(f"{verdict1} vs {verdict2}")
        if verdict1 == verdict2:
            same_verdicts += 1
        else:
            different_verdicts += 1
    return [same_verdicts, different_verdicts]


def print_results(verdicts):
    print("=" * 100)
    total_verdicts = verdicts[0] + verdicts[1]
    print(f"Same verdicts: {verdicts[0]}")
    print(f"Different verdicts: {verdicts[1]}")
    accuracy = verdicts[0] / total_verdicts
    print(f"Algorithm accuracy: {accuracy*100:.2f}%")
    if accuracy > 0.9:
        print(":)")
    else:
        print(":(")
    print("="*100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--step_length", type=float, default=0.001)
    parser.add_argument("--excluded_columns", type=str, default="")
    args = parser.parse_args()
    excluded_columns = []
    for column in args.excluded_columns.split(","):
        excluded_columns.append(column)

    X, y = get_data(excluded_columns)
    data = prepare_data(X, y)
    test_diagnosis = logistic_regression(data, args.iterations, args.step_length)
    #print(f"Real diagnosis: {data[3].values.tolist()}")
    #print(f"Algorithm diagnosis: {test_diagnosis}")
    test_results = compare_diagnoses(data[3], test_diagnosis)
    print_results(test_results)


if __name__ == "__main__":
    main()