from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(chosen_columns: list):
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    data = X.copy()
    data['diagnosis'] = y
    data = data.dropna()
    X = data[chosen_columns]
    y = data['diagnosis']

    # convert string values to binary for our algorithm
    for i in range(len(y)):
        if y[i] == 'M':
            y[i] = 1
        elif y[i] == 'B':
            y[i] = 0

    return X, y


def prepare_data(test_ratio: float=0.25):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=69)
    return X_train, X_test, y_train, y_test