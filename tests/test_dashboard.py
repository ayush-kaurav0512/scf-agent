"""Tests for the dashboard fetch helpers and the chat-append logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import streamlit as st

from dashboard.components.irr_chart import _fetch_irr, _fetch_whatif
from dashboard.components.nl_panel import _append_turn
from dashboard.components.risk_table import _fetch_at_risk


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    """Streamlit's cache_data persists across tests; clear it between runs."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _mock_response(json_payload: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=json_payload)
    if status_code >= 400:
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "boom",
                request=MagicMock(),
                response=response,
            )
        )
    else:
        response.raise_for_status = MagicMock(return_value=None)
    return response


# ---------------------------------------------------------------------------
# _fetch_at_risk
# ---------------------------------------------------------------------------

def test_fetch_at_risk_returns_dict() -> None:
    fake_payload = {
        "at_risk_count": 2,
        "threshold": 50,
        "suppliers": [
            {"supplier_name": "alpha", "credit_score": 30.0, "distress_prob": 0.7,
             "risk_label": "high_risk", "summary": "alpha 30/100 (high_risk)."},
            {"supplier_name": "beta",  "credit_score": 45.0, "distress_prob": 0.55,
             "risk_label": "high_risk", "summary": "beta 45/100 (high_risk)."},
        ],
        "summary": "2 suppliers are below score 50.",
    }

    with patch("httpx.get", return_value=_mock_response(fake_payload)):
        result = _fetch_at_risk("http://localhost:8000", 50)

    assert set(result.keys()) >= {"at_risk_count", "suppliers", "summary"}
    assert result["at_risk_count"] == 2
    assert len(result["suppliers"]) == 2


def test_fetch_at_risk_handles_http_error() -> None:
    with patch("httpx.get", return_value=_mock_response({"detail": "boom"}, status_code=500)):
        result = _fetch_at_risk("http://localhost:8000", 50)

    assert "error" in result


# ---------------------------------------------------------------------------
# _fetch_irr
# ---------------------------------------------------------------------------

def test_irr_response_recommendation_values() -> None:
    fake_irr = {
        "irr": 0.73,
        "npv": 1885.0,
        "discount_captured": 2000.0,
        "wacc": 0.084,
        "profitable": True,
        "credit_ok": True,
        "recommendation": "PAY_EARLY",
        "summary": "Recommendation: PAY_EARLY.",
    }

    with patch("httpx.post", return_value=_mock_response(fake_irr)):
        result = _fetch_irr(
            "http://localhost:8000",
            {
                "invoice_value": 100_000.0,
                "credit_score": 80.0,
                "days_early": 10,
                "discount_pct": 2.0,
            },
        )

    assert result["recommendation"] in {"PAY_EARLY", "HOLD", "FLAG"}


# ---------------------------------------------------------------------------
# _fetch_whatif
# ---------------------------------------------------------------------------

def test_whatif_response_has_scenarios() -> None:
    scenarios = [
        {
            "days_early": days,
            "discount_pct": pct,
            "irr": 0.1,
            "npv": 100.0,
            "discount_captured": 50.0,
            "profitable": True,
            "recommendation": "PAY_EARLY",
        }
        for days in (5, 10, 15, 20, 30)
        for pct in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    ]
    fake_whatif = {
        "total_scenarios": 30,
        "scenarios": scenarios,
        "top_opportunities": scenarios[:5],
        "summary": {"total_scenarios": 30, "profitable_count": 30, "best_irr": 0.1,
                    "best_recommendation": "PAY_EARLY", "avg_npv": 100.0},
    }

    with patch("httpx.post", return_value=_mock_response(fake_whatif)):
        result = _fetch_whatif(
            "http://localhost:8000",
            {"invoice_value": 100_000.0, "credit_score": 80.0},
        )

    assert len(result["scenarios"]) == 30


# ---------------------------------------------------------------------------
# _append_turn (chat-history mutation)
# ---------------------------------------------------------------------------

def test_nl_panel_appends_messages() -> None:
    messages: list = []
    _append_turn(messages, "user", "Who is low risk?")
    _append_turn(messages, "assistant", "PrimeParts is low risk.")

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[0]["content"] == "Who is low risk?"
    assert messages[1]["content"] == "PrimeParts is low risk."
