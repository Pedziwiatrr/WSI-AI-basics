import pandas as pd


CSV_PATH = "us-crime-dataset/US_Crime_DataSet.csv"
FEATURES =  [
    "Victim Sex", "Victim Age", "Victim Race", "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race", "Relationship", "Weapon"
]


def prepare_data(csv_file=CSV_PATH, selected_features=FEATURES):
    data = pd.read_csv(csv_file)
    data = data[selected_features]
    prepared_data = data.dropna()

    return prepared_data

