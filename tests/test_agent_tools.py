"""Tests for the LangChain tools exposed to the SCF agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scf_agent.agent.tools import (
    get_at_risk_suppliers,
    get_macro_indicators,
    get_supplier_risk_score,
    run_irr_simulation,
    run_whatif_simulation,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def features_df() -> pd.DataFrame:
    """Synthetic supplier feature frame used by the scorer tools."""
    return pd.DataFrame(
        {
            "supplier_name": ["acme corp", "beta llc", "gamma inc"],
            "lead_time_variance": [1.0, 5.0, 2.0],
            "return_rate": [0.01, 0.05, 0.02],
            "late_delivery_rate": [0.1, 0.7, 0.2],
            "avg_payment_delay": [1.0, 4.0, 2.0],
            "order_volume": [100, 200, 150],
            "avg_order_value": [120.0, 300.0, 180.0],
        }
    )


def _scored_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["supplier_name", "credit_score", "distress_prob", "risk_label"])


# ---------------------------------------------------------------------------
# get_supplier_risk_score
# ---------------------------------------------------------------------------

def test_get_supplier_risk_score_not_found(features_df: pd.DataFrame) -> None:
    mock_scorer = MagicMock()
    mock_scorer.predict_scores.return_value = _scored_frame([])

    with patch("scf_agent.models.risk_scorer.SupplierRiskScorer.load", return_value=mock_scorer), \
         patch("pandas.read_parquet", return_value=features_df):
        result = get_supplier_risk_score.invoke({"supplier_name": "ghost industries"})

    assert "error" in result
    assert "ghost industries" in result["error"]
    mock_scorer.predict_scores.assert_not_called()


def test_get_supplier_risk_score_found(features_df: pd.DataFrame) -> None:
    mock_scorer = MagicMock()
    mock_scorer.predict_scores.return_value = _scored_frame(
        [{"supplier_name": "acme corp", "credit_score": 88.5, "distress_prob": 0.115, "risk_label": "low_risk"}]
    )

    with patch("scf_agent.models.risk_scorer.SupplierRiskScorer.load", return_value=mock_scorer), \
         patch("pandas.read_parquet", return_value=features_df):
        result = get_supplier_risk_score.invoke({"supplier_name": "  ACME Corp "})

    assert result["supplier_name"] == "acme corp"
    assert result["credit_score"] == pytest.approx(88.5)
    assert result["risk_label"] == "low_risk"
    assert "88.5/100" in result["summary"]


# ---------------------------------------------------------------------------
# get_at_risk_suppliers
# ---------------------------------------------------------------------------

def test_get_at_risk_suppliers_returns_correct_keys(features_df: pd.DataFrame) -> None:
    mock_scorer = MagicMock()
    mock_scorer.predict_scores.return_value = _scored_frame(
        [
            {"supplier_name": "acme corp", "credit_score": 82.0, "distress_prob": 0.18, "risk_label": "low_risk"},
            {"supplier_name": "beta llc",  "credit_score": 30.0, "distress_prob": 0.70, "risk_label": "high_risk"},
            {"supplier_name": "gamma inc", "credit_score": 45.0, "distress_prob": 0.55, "risk_label": "high_risk"},
        ]
    )

    with patch("scf_agent.models.risk_scorer.SupplierRiskScorer.load", return_value=mock_scorer), \
         patch("pandas.read_parquet", return_value=features_df):
        result = get_at_risk_suppliers.invoke({"threshold": 50})

    assert set(result.keys()) == {"at_risk_count", "threshold", "suppliers", "summary"}
    assert result["at_risk_count"] == 2
    assert result["threshold"] == 50
    assert [s["supplier_name"] for s in result["suppliers"]] == ["beta llc", "gamma inc"]
    assert "2 suppliers" in result["summary"]


# ---------------------------------------------------------------------------
# run_irr_simulation
# ---------------------------------------------------------------------------

def test_run_irr_simulation_pay_early() -> None:
    result = run_irr_simulation.invoke(
        {
            "days_early": 10,
            "discount_pct": 2.0,
            "invoice_value": 100_000.0,
            "credit_score": 80.0,
        }
    )

    assert result["recommendation"] == "PAY_EARLY"
    assert "summary" in result
    assert "PAY_EARLY" in result["summary"]


def test_run_irr_simulation_flag() -> None:
    result = run_irr_simulation.invoke(
        {
            "days_early": 10,
            "discount_pct": 2.0,
            "invoice_value": 100_000.0,
            "credit_score": 20.0,
        }
    )

    assert result["recommendation"] == "FLAG"


# ---------------------------------------------------------------------------
# get_macro_indicators
# ---------------------------------------------------------------------------

def test_get_macro_indicators_keys() -> None:
    fake = {
        "fed_funds_rate": 5.33,
        "cpi_inflation": 312.5,
        "treasury_10y": 4.28,
        "wacc": 0.084,
    }

    with patch("scf_agent.pipeline.ingest.FREDFetcher.get_indicators", return_value=fake):
        result = get_macro_indicators.invoke({})

    assert set(result.keys()) == {"fed_funds_rate", "cpi_inflation", "treasury_10y", "wacc", "summary"}
    assert result["fed_funds_rate"] == pytest.approx(5.33)
    assert "5.33" in result["summary"]


# ---------------------------------------------------------------------------
# run_whatif_simulation
# ---------------------------------------------------------------------------

def test_run_whatif_simulation_empty_on_zero_score() -> None:
    result = run_whatif_simulation.invoke(
        {
            "invoice_value": 100_000.0,
            "credit_score": 0.0,
        }
    )

    assert result["top_opportunities"] == []
    assert "No profitable scenarios" in result["summary"]
