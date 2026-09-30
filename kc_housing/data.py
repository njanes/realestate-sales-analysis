"""Loading and cleaning the King County house sales data."""

import pandas as pd

from .config import DATA_PATH, FEATURES, TARGET

DATE_FORMAT = "%Y%m%dT%H%M%S"
MEAN_IMPUTED = ["bedrooms", "bathrooms"]


def load_raw(path=DATA_PATH):
    """Read the sales CSV as downloaded."""
    return pd.read_csv(path)


def clean(df):
    """Return a cleaned copy of the raw data.

    Drops the id column, parses the sale date, and fills any missing bedroom
    or bathroom counts with the column mean. The Kaggle copy of the dataset
    has no missing values, so the imputation only matters for copies of the
    file that do.
    """
    df = df.drop(columns=["id"])
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
    for col in MEAN_IMPUTED:
        df[col] = df[col].fillna(df[col].mean())
    return df


def load_clean(path=DATA_PATH):
    """Read and clean the dataset in one step."""
    return clean(load_raw(path))


def features_and_target(df):
    """Split a cleaned frame into the model features (X) and sale price (y)."""
    return df[FEATURES], df[TARGET]
