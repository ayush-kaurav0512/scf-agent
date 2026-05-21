"""API tests using FastAPI's TestClient with all heavy deps mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from scf_agent.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert isinstance(body["model_loaded"], bool)


# ---------------------------------------------------------------------------
# /api/v1/suppliers
# ---------------------------------------------------------------------------

def test_get_supplier_found() -> None:
    fake = {
        "supplier_name": "primeparts",
        "credit_score": 88.5,
        "distress_prob": 0.115,
        "risk_label": "low_risk",
        "summary": "primeparts has a credit score of 88.5/100 (low_risk).",
    }

    with patch("scf_agent.api.routers.suppliers.get_supplier_risk_score") as mock_tool:
        mock_tool.invoke.return_value = fake
        response = client.get("/api/v1/suppliers/PrimeParts")

    assert response.status_code == 200
    data = response.json()["data"]
    assert 0.0 <= data["credit_score"] <= 100.0
    assert data["risk_label"] in {"low_risk", "watch", "high_risk"}


def test_get_supplier_not_found() -> None:
    with patch("scf_agent.api.routers.suppliers.get_supplier_risk_score") as mock_tool:
        mock_tool.invoke.return_value = {"error": "Supplier not found: Unknown"}
        response = client.get("/api/v1/suppliers/Unknown")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /api/v1/decisions/irr
# ---------------------------------------------------------------------------

def test_calculate_irr_pay_early() -> None:
    fake = {
        "irr": 0.73,
        "npv": 1885.0,
        "discount_captured": 2000.0,
        "wacc": 0.084,
        "profitable": True,
        "credit_ok": True,
        "recommendation": "PAY_EARLY",
        "summary": "Paying $100,000 10 days early ... Recommendation: PAY_EARLY.",
    }

    with patch("scf_agent.api.routers.decisions.run_irr_simulation") as mock_tool:
        mock_tool.invoke.return_value = fake
        response = client.post(
            "/api/v1/decisions/irr",
            json={
                "days_early": 10,
                "discount_pct": 2.0,
                "invoice_value": 100_000.0,
                "credit_score": 80.0,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["recommendation"] == "PAY_EARLY"
    assert body["irr"] == 0.73


def test_calculate_irr_invalid_body() -> None:
    response = client.post(
        "/api/v1/decisions/irr",
        json={
            "days_early": 0,
            "discount_pct": 2.0,
            "invoice_value": 100_000.0,
            "credit_score": 80.0,
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /api/v1/decisions/macro
# ---------------------------------------------------------------------------

def test_macro_indicators() -> None:
    fake = {
        "fed_funds_rate": 5.33,
        "cpi_inflation": 3.10,
        "treasury_10y": 4.28,
        "wacc": 0.084,
        "summary": "Fed Funds Rate: 5.33%, CPI: 3.10, 10Y Treasury: 4.28%, WACC: 8.4%.",
    }

    with patch("scf_agent.api.routers.decisions.get_macro_indicators") as mock_tool:
        mock_tool.invoke.return_value = fake
        response = client.get("/api/v1/decisions/macro")

    assert response.status_code == 200
    body = response.json()
    assert "fed_funds_rate" in body
    assert body["fed_funds_rate"] == 5.33


# ---------------------------------------------------------------------------
# /api/v1/agent/ask
# ---------------------------------------------------------------------------

def test_ask_agent() -> None:
    async_mock = AsyncMock(return_value="PrimeParts is low risk.")

    with patch("scf_agent.api.routers.agent.ask", new=async_mock):
        response = client.post(
            "/api/v1/agent/ask",
            json={"question": "Who is low risk?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "PrimeParts is low risk."
    assert body["question"] == "Who is low risk?"
    async_mock.assert_awaited_once_with("Who is low risk?")
