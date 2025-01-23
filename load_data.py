import pandas as pd


CSV_PATH = "us-crime-dataset/US_Crime_DataSet.csv"


def view_data():
    data = pd.read_csv(CSV_PATH)
    print(data.head())