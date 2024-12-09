from load_data import get_data, prepare_data


def main():
    X, y = get_data([])
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    print(X_train, X_test, y_train, y_test)
    print(X_train.shape)


if __name__ == "__main__":
    main()