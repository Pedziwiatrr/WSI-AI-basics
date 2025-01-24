import pandas as pd
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "us-crime-dataset/US_Crime_DataSet.csv"
FEATURES =  [
    "Victim Sex", "Victim Age", "Victim Race", "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race", "Relationship", "Weapon"
]


def get_prepared_data(csv_file=CSV_PATH, selected_features=FEATURES):
    data = pd.read_csv(csv_file, low_memory=False)
    data = data[selected_features]
    data = data[~data.isin(['Unknown', 'unknown']).any(axis=1)]
    prepared_data = data.dropna()

    return prepared_data


def get_numerical_data(prepared_data=get_prepared_data()):
    encoded_data = prepared_data.copy()

    for column in prepared_data.columns:
        encoder = LabelEncoder()
        encoded_data[column] = encoder.fit_transform(prepared_data[column])

    return encoded_data