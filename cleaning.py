import numpy as np
import pandas as pd

DATA_PATH = "data/kc_house_data.csv"


def clean(data):
    data = data.drop(columns=["id"])
    data["date"] = pd.to_datetime(data["date"], format="%Y%m%dT%H%M%S")

    # 0 bedrooms or bathrooms, and the 33-bedroom typo, are treated as missing
    data["bedrooms"] = data["bedrooms"].replace({0: np.nan, 33: np.nan})
    data["bathrooms"] = data["bathrooms"].replace(0, np.nan)
    data["bedrooms"] = data["bedrooms"].fillna(data["bedrooms"].median())
    data["bathrooms"] = data["bathrooms"].fillna(data["bathrooms"].median())
    return data
