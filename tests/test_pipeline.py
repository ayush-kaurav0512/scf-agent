"""Tests for the DataCo ingester, FRED fetcher, DataCo preprocessor, and feature engineer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scf_agent.config import settings
from scf_agent.pipeline.features import FeatureEngineer
from scf_agent.pipeline.ingest import (
    REQUIRED_DATACO_COLUMNS,
    DataCoIngester,
    FREDFetcher,
)
from scf_agent.pipeline.preprocess import Preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_dataco_df() -> pd.DataFrame:
    """A DataCo-shaped DataFrame with >= 1000 rows."""
    n = 1500
    return pd.DataFrame(
        {
            "order_id": range(n),
            "customer_id": [1000 + i for i in range(n)],
            "order_date": ["2026-01-01"] * n,
            "ship_date": ["2026-01-05"] * n,
            "delivery_status": ["Shipping on time"] * n,
            "late_delivery_risk": [0] * n,
            "order_item_quantity": [1] * n,
            "order_item_total": [99.99] * n,
            "supplier_name": ["Acme Corp"] * n,
            "product_category": ["Electronics"] * n,
        }
    )


@pytest.fixture
def small_dataco_df(valid_dataco_df: pd.DataFrame) -> pd.DataFrame:
    """A DataCo-shaped frame that has fewer than the required 1000 rows."""
    return valid_dataco_df.head(50).copy()


@pytest.fixture
def preprocessor_df() -> pd.DataFrame:
    """A small frame exercising every Preprocessor step."""
    return pd.DataFrame(
        {
            "order_id": [1, 1, 2, 3],
            "customer_id": [10, 10, 20, 30],
            "order_date": ["2026-01-01", "2026-01-01", "2026-02-15", "2026-03-10"],
            "ship_date":  ["2026-01-05", "2026-01-05", "2026-02-10", "2026-03-12"],
            "delivery_status": ["  Shipping  ", "  Shipping  ", "LATE", "ON TIME"],
            "supplier_name":   ["  Acme Corp ", "  Acme Corp ", "BETA LLC", "Gamma Inc"],
            "product_category": ["Electronics", "Electronics", "Apparel", "Books"],
            "late_delivery_risk":   [0, 0, 1, None],
            "order_item_quantity":  [1, 1, 2, 3],
            "order_item_total":     [99.0, 99.0, None, 250.0],
        }
    )


# ---------------------------------------------------------------------------
# DataCoIngester.validate
# ---------------------------------------------------------------------------

def test_validate_raises_value_error_on_missing_columns(valid_dataco_df: pd.DataFrame) -> None:
    df = valid_dataco_df.drop(columns=["supplier_name", "product_category"])
    ingester = DataCoIngester(raw_path=Path("/tmp/unused.csv"))

    with pytest.raises(ValueError, match="missing required columns"):
        ingester.validate(df)


def test_validate_raises_value_error_when_row_count_below_minimum(small_dataco_df: pd.DataFrame) -> None:
    ingester = DataCoIngester(raw_path=Path("/tmp/unused.csv"))

    with pytest.raises(ValueError, match="at least 1000 required"):
        ingester.validate(small_dataco_df)


def test_validate_returns_df_when_valid(valid_dataco_df: pd.DataFrame) -> None:
    ingester = DataCoIngester(raw_path=Path("/tmp/unused.csv"))
    out = ingester.validate(valid_dataco_df)
    assert out is valid_dataco_df
    assert all(col in out.columns for col in REQUIRED_DATACO_COLUMNS)


def test_load_raises_file_not_found_for_missing_path() -> None:
    ingester = DataCoIngester(raw_path=Path("/tmp/does_not_exist_scf_agent.csv"))

    with pytest.raises(FileNotFoundError, match="DataCo CSV not found"):
        ingester.load()


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def test_add_lead_time_produces_non_negative_values(preprocessor_df: pd.DataFrame) -> None:
    out = (
        Preprocessor(preprocessor_df)
        .drop_duplicates()
        .parse_dates()
        .add_lead_time()
        .df
    )

    assert "lead_time_days" in out.columns
    assert (out["lead_time_days"] >= 0).all()
    assert out.loc[out["order_id"] == 2, "lead_time_days"].iloc[0] == 0


def test_clean_strings_lowercases_supplier_name(preprocessor_df: pd.DataFrame) -> None:
    out = Preprocessor(preprocessor_df).clean_strings().df

    suppliers = out["supplier_name"].tolist()
    assert all(s == s.lower() for s in suppliers)
    assert "acme corp" in suppliers
    assert "beta llc" in suppliers


# ---------------------------------------------------------------------------
# FREDFetcher.get_indicators (mocked HTTP)
# ---------------------------------------------------------------------------

def _mock_fred_response(value: str) -> MagicMock:
    """Build a MagicMock that quacks like an httpx.Response."""
    response = MagicMock()
    response.raise_for_status = MagicMock(return_value=None)
    response.json = MagicMock(return_value={"observations": [{"date": "2026-05-01", "value": value}]})
    return response


def test_get_indicators_returns_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "FRED_API_KEY", "fake-key")
    monkeypatch.setattr(settings, "WACC", 0.084)

    series_to_value = {
        "FEDFUNDS": "5.25",
        "CPIAUCSL": "312.0",
        "DGS10":    "4.40",
    }

    def _fake_get(self, url, params=None):  # noqa: ANN001 — matches httpx.Client.get
        series_id = params["series_id"]
        return _mock_fred_response(series_to_value[series_id])

    with patch("scf_agent.pipeline.ingest.httpx.Client") as mock_client_cls:
        client_instance = MagicMock()
        client_instance.get = MagicMock(side_effect=lambda url, params=None: _fake_get(client_instance, url, params))
        mock_client_cls.return_value.__enter__.return_value = client_instance

        indicators = FREDFetcher().get_indicators()

    assert set(indicators.keys()) == {"fed_funds_rate", "cpi_inflation", "treasury_10y", "wacc"}
    assert indicators["fed_funds_rate"] == pytest.approx(5.25)
    assert indicators["cpi_inflation"] == pytest.approx(312.0)
    assert indicators["treasury_10y"] == pytest.approx(4.40)
    assert indicators["wacc"] == pytest.approx(0.084)


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_input_df() -> pd.DataFrame:
    """Synthetic clean DataCo frame: 3 suppliers x 10 orders each."""
    suppliers = ["acme corp", "beta llc", "gamma inc"]
    rows: list[dict] = []
    order_id = 0
    for s_idx, supplier in enumerate(suppliers):
        for i in range(10):
            order_id += 1
            if i % 5 == 0:
                status = "returned"
            elif i % 3 == 0:
                status = "late delivery"
            else:
                status = "shipping on time"
            rows.append(
                {
                    "order_id": order_id,
                    "supplier_name": supplier,
                    "order_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    "ship_date":  pd.Timestamp("2026-01-01") + pd.Timedelta(days=i + 3 + s_idx),
                    "delivery_status": status,
                    "late_delivery_risk": 1 if i % 2 == 0 else 0,
                    "order_item_quantity": 1 + (i % 4),
                    "order_item_total": 50.0 + 10.0 * i + 5.0 * s_idx,
                    "lead_time_days": (3 + s_idx + (i % 4)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def no_disk_writes(monkeypatch: pytest.MonkeyPatch):
    """Stub out parquet writes and ``mkdir`` so tests stay in memory."""
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *a, **kw: None)
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **kw: None)


def test_lead_time_variance_nonnegative(feature_input_df: pd.DataFrame, no_disk_writes) -> None:
    features = FeatureEngineer(feature_input_df).build()
    assert "lead_time_variance" in features.columns
    assert (features["lead_time_variance"] >= 0).all()


def test_return_rate_bounded(feature_input_df: pd.DataFrame, no_disk_writes) -> None:
    features = FeatureEngineer(feature_input_df).build()
    assert "return_rate" in features.columns
    assert features["return_rate"].between(0.0, 1.0, inclusive="both").all()


def test_late_delivery_rate_bounded(feature_input_df: pd.DataFrame, no_disk_writes) -> None:
    features = FeatureEngineer(feature_input_df).build()
    assert "late_delivery_rate" in features.columns
    assert features["late_delivery_rate"].between(0.0, 1.0, inclusive="both").all()


def test_feature_matrix_no_nulls(feature_input_df: pd.DataFrame, no_disk_writes) -> None:
    features = FeatureEngineer(feature_input_df).build()
    assert features.isna().sum().sum() == 0


def test_supplier_name_is_column(feature_input_df: pd.DataFrame, no_disk_writes) -> None:
    features = FeatureEngineer(feature_input_df).build()
    assert "supplier_name" in features.columns
    assert features.index.name != "supplier_name"
    assert set(features["supplier_name"]) == {"acme corp", "beta llc", "gamma inc"}
