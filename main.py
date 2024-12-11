import argparse
from load_data import get_data, prepare_data
from logistic_regression import logistic_regression


def compare_diagnoses(first_diagnosis, second_diagnosis):
    same_verdicts = 0
    different_verdicts = 0
    for verdict in first_diagnosis:
        if verdict == second_diagnosis[verdict]:
            same_verdicts += 1
        else:
            different_verdicts += 1
    return [same_verdicts, different_verdicts]


def print_results(verdicts):
    total_verdicts = verdicts[0] + verdicts[1]
    print(f"Same verdicts: {verdicts[0]}")
    print(f"Different verdicts: {verdicts[1]}")
    print(f"Algorithm accuracy: {verdicts[0] / total_verdicts}")

def main():
    X, y = get_data([])
    data = prepare_data(X, y)
    test_diagnosis = logistic_regression(data, 500, 0.001)
    test_results = compare_diagnoses(data[3], test_diagnosis)
    print_results(test_results)



if __name__ == "__main__":
    main()