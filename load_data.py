from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


def get_data():
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # metadata
    print(wine_quality.metadata)

    # variable information
    print(wine_quality.variables)

    return X, y


def prepare_data(X, y, test_ratio: float=0.25, seed=69):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
    return [X_train, X_test, y_train, y_test]