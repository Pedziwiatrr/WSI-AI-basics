from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_data():
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    return X, y


def prepare_data(X, y, test_ratio: float=0.25, seed=69):
    print(X.isnull().sum())
    X = X.fillna(X.mean())  # if any value is missing its replaced by mean of other values of this feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # scaling data
    # mean = 0, variance = 1
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_ratio, random_state=seed)
    return [X_train, X_test, y_train, y_test]
