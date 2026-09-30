import numpy as np
import pandas as pd
import pytest

from kc_housing.data import clean, load_clean


@pytest.fixture
def raw():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "date": ["20141013T000000", "20141209T000000", "20150225T000000", "20150218T000000"],
        "price": [221900.0, 538000.0, 180000.0, 604000.0],
        "bedrooms": [3.0, np.nan, 2.0, 4.0],
        "bathrooms": [1.0, 2.25, np.nan, 3.0],
    })


def test_drops_id(raw):
    assert "id" not in clean(raw).columns


def test_parses_date(raw):
    out = clean(raw)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["date"].iloc[0] == pd.Timestamp("2014-10-13")


def test_fills_missing_with_column_mean(raw):
    out = clean(raw)
    assert out[["bedrooms", "bathrooms"]].notna().all().all()
    assert out.loc[1, "bedrooms"] == pytest.approx((3 + 2 + 4) / 3)
    assert out.loc[2, "bathrooms"] == pytest.approx((1 + 2.25 + 3) / 3)


def test_does_not_modify_input(raw):
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_load_clean_reads_csv(raw, tmp_path):
    path = tmp_path / "sales.csv"
    raw.to_csv(path, index=False)
    out = load_clean(path)
    assert len(out) == len(raw)
    assert "id" not in out.columns
