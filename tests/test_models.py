import numpy as np
import pandas as pd
import pytest

from kc_housing.config import FEATURES, N_SPLITS, TARGET
from kc_housing.evaluate import cv_r2, results_table, to_markdown
from kc_housing.models import poly_linear, ridge


@pytest.fixture
def sales():
    """Random data with the real column names and a price that depends on them."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, len(FEATURES))), columns=FEATURES)
    df[TARGET] = 3 * df["sqft_living"] + df["grade"] ** 2 + rng.normal(0, 0.1, 200)
    return df


def test_ridge_scales_before_fitting():
    assert [name for name, _ in ridge().steps] == ["scale", "model"]


def test_ridge_polynomial_step():
    model = ridge(alpha=2.0, degree=2)
    assert [name for name, _ in model.steps] == ["scale", "polynomial", "model"]
    assert model.named_steps["polynomial"].include_bias is False
    assert model.named_steps["model"].alpha == 2.0


def test_poly_linear_matches_ridge_preprocessing():
    assert [name for name, _ in poly_linear().steps] == ["scale", "polynomial", "model"]


def test_cv_uses_the_same_folds_every_call(sales):
    X, y = sales[FEATURES], sales[TARGET]
    first = cv_r2(ridge(), X, y)
    second = cv_r2(ridge(), X, y)
    assert len(first) == N_SPLITS
    np.testing.assert_array_equal(first, second)


def test_results_table_shape(sales):
    table = results_table(sales)
    assert len(table) == 7
    assert table["R2 std"].notna().sum() == 2  # only the CV rows have a spread


def test_markdown_bolds_best_cv_score(sales):
    md = to_markdown(results_table(sales))
    assert md.splitlines()[0] == "|Model|Features|R²|Evaluated on|"
    assert md.count("**") == 2
