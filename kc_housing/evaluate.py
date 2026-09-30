"""Scoring helpers and the summary results table."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from .config import ALPHA, FEATURES, N_SPLITS, RANDOM_STATE, TEST_SIZE
from .data import features_and_target
from .models import poly_linear, ridge


def split(X, y):
    """The fixed train/test split used throughout the analysis."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_r2(model, X, y):
    """Fit on all the data and score on the same data (in-sample R-squared)."""
    return model.fit(X, y).score(X, y)


def holdout_r2(model, X_train, X_test, y_train, y_test):
    """Fit on the training split and score on the test split."""
    return model.fit(X_train, y_train).score(X_test, y_test)


def cv_r2(model, X, y, n_splits=N_SPLITS):
    """R-squared for each cross-validation fold.

    The folds are shuffled with a fixed seed, so every call uses the same
    folds and scores from different models can be compared fold by fold.
    """
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=folds, scoring="r2")


def results_table(df):
    """Fit every model in the analysis and collect their scores.

    Returns one row per model with the mean R-squared and, for
    cross-validated rows, its standard deviation across folds.
    """
    X, y = features_and_target(df)
    X_train, X_test, y_train, y_test = split(X, y)
    all_features = f"{len(FEATURES)} features"
    on_train = "Training data"
    on_test = f"Test set ({TEST_SIZE:.0%})"
    on_cv = f"{N_SPLITS}-fold cross-validation"
    ridge_name = f"Ridge (α = {ALPHA})"
    ridge_poly_name = f"{ridge_name} + degree-2 polynomial"

    runs = [
        ("Linear regression", "`sqft_living` only", on_train,
         train_r2(LinearRegression(), df[["sqft_living"]], y)),
        ("Linear regression", all_features, on_train,
         train_r2(LinearRegression(), X, y)),
        ("Pipeline (scaling + polynomial + linear)", all_features, on_train,
         train_r2(poly_linear(), X, y)),
        (ridge_name, all_features, on_test,
         holdout_r2(ridge(), X_train, X_test, y_train, y_test)),
        (ridge_poly_name, all_features, on_test,
         holdout_r2(ridge(degree=2), X_train, X_test, y_train, y_test)),
        (ridge_name, all_features, on_cv, cv_r2(ridge(), X, y)),
        (ridge_poly_name, all_features, on_cv, cv_r2(ridge(degree=2), X, y)),
    ]

    rows = []
    for model, features, evaluated_on, scores in runs:
        scores = np.atleast_1d(scores)
        rows.append({
            "Model": model,
            "Features": features,
            "Evaluated on": evaluated_on,
            "R2": scores.mean(),
            "R2 std": scores.std() if len(scores) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def format_table(table):
    """Readable version of results_table: one R² column, with ± std for CV rows."""
    def fmt(row):
        if pd.isna(row["R2 std"]):
            return f"{row['R2']:.2f}"
        return f"{row['R2']:.3f} ± {row['R2 std']:.3f}"

    out = table[["Model", "Features"]].copy()
    out["R²"] = table.apply(fmt, axis=1)
    out["Evaluated on"] = table["Evaluated on"]
    return out


def to_markdown(table):
    """Markdown version of the table for the README, with the best CV score in bold."""
    shown = format_table(table)
    cv_rows = table["R2 std"].notna()
    if cv_rows.any():
        best = table.loc[cv_rows, "R2"].idxmax()
        shown.loc[best, "R²"] = f"**{shown.loc[best, 'R²']}**"

    lines = [
        "|" + "|".join(shown.columns) + "|",
        "|" + "|".join("-" for _ in shown.columns) + "|",
    ]
    for _, row in shown.iterrows():
        lines.append("|" + "|".join(str(v) for v in row) + "|")
    return "\n".join(lines)
