import argparse
from load_data import get_data, prepare_data


def main():
    X, y = get_data()
    data = prepare_data(X, y, test_ratio=0.25, seed=69)


if __name__ == '__main__':
    main()