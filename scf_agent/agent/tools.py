"""LangChain tools exposed to the SCF agent.

Each tool is a thin, JSON-safe wrapper around the domain layer
(:mod:`scf_agent.models.risk_scorer`, :mod:`scf_agent.optimization.irr_engine`,
:mod:`scf_agent.pipeline.ingest`).

Two invariants:

* **Lazy dependencies** — heavy modules (scorer, IRR engine, FRED client) are
  imported inside the tool body so a) startup stays fast and b) we avoid any
  chance of a circular import.
* **Never raise** — every tool catches its own exceptions and returns
  ``{"error": str}`` so the agent can reason over the failure instead of
  crashing the executor.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from scf_agent.config import settings

logger = logging.getLogger(__name__)

FEATURES_FILENAME: str = "features.parquet"


def _load_scored_features() -> "tuple[object, object]":
    """Lazy helper: load the trained scorer and the supplier feature frame."""
    import pandas as pd

    from scf_agent.models.risk_scorer import SupplierRiskScorer

    scorer = SupplierRiskScorer.load()
    features_path = settings.DATA_PROCESSED_DIR / FEATURES_FILENAME
    features = pd.read_parquet(features_path)
    return scorer, features


@tool
def get_supplier_risk_score(supplier_name: str) -> dict:
    """Look up the current credit score and risk label for a single supplier
    by name. Returns credit_score (0-100), distress_prob, risk_label
    (low_risk / watch / high_risk), and a plain-English summary.
    Use this when the user asks about a specific supplier's financial health.
    """
    try:
        scorer, features = _load_scored_features()

        normalized = supplier_name.strip().lower()
        names = features["supplier_name"].astype(str).str.strip().str.lower()
        match = features[names == normalized]

        if match.empty:
            return {"error": f"Supplier not found: {supplier_name}"}

        scored = scorer.predict_scores(match)
        row = scored.iloc[0]
        credit_score = float(row["credit_score"])
        risk_label = str(row["risk_label"])
        resolved_name = str(row["supplier_name"])

        return {
            "supplier_name": resolved_name,
            "credit_score": credit_score,
            "distress_prob": float(row["distress_prob"]),
            "risk_label": risk_label,
            "summary": f"{resolved_name} has a credit score of {credit_score}/100 ({risk_label}).",
        }
    except Exception as exc:
        logger.exception("get_supplier_risk_score failed")
        return {"error": str(exc)}


@tool
def get_at_risk_suppliers(threshold: int = 50) -> dict:
    """Return all suppliers whose credit score is below the given threshold.
    Default threshold is 50 (watch + high_risk combined). Returns a list of
    suppliers sorted by credit_score ascending (worst first), plus a
    plain-English summary of how many are at risk.
    Use this when the user asks for an overview of risky suppliers.
    """
    try:
        scorer, features = _load_scored_features()
        scored = scorer.predict_scores(features)

        at_risk = scored[scored["credit_score"] < threshold].sort_values(
            "credit_score", ascending=True
        )

        suppliers = []
        for _, row in at_risk.iterrows():
            name = str(row["supplier_name"])
            score = float(row["credit_score"])
            label = str(row["risk_label"])
            suppliers.append(
                {
                    "supplier_name": name,
                    "credit_score": score,
                    "distress_prob": float(row["distress_prob"]),
                    "risk_label": label,
                    "summary": f"{name} has a credit score of {score}/100 ({label}).",
                }
            )

        count = len(suppliers)
        return {
            "at_risk_count": count,
            "threshold": int(threshold),
            "suppliers": suppliers,
            "summary": f"{count} suppliers are below score {threshold}.",
        }
    except Exception as exc:
        logger.exception("get_at_risk_suppliers failed")
        return {"error": str(exc)}


@tool
def run_irr_simulation(
    days_early: int,
    discount_pct: float,
    invoice_value: float,
    credit_score: float = 75.0,
) -> dict:
    """Calculate the annualised IRR and NPV of a proposed early payment, and
    return a PAY_EARLY / HOLD / FLAG recommendation. Inputs: days_early (int),
    discount_pct (float, e.g. 2.0 for 2%), invoice_value (float, USD),
    credit_score (float 0-100, default 75).
    Use this when the user asks whether it is profitable to pay a supplier
    early in exchange for a discount.
    """
    try:
        from scf_agent.optimization.irr_engine import IRRCalculator

        calc = IRRCalculator()
        decision = calc.decision(
            discount_pct=discount_pct,
            days_early=days_early,
            invoice_value=invoice_value,
            credit_score=credit_score,
        )

        irr_pct = decision["irr"] * 100.0
        wacc_pct = decision["wacc"] * 100.0
        summary = (
            f"Paying ${invoice_value:,.0f} {days_early} days early for a "
            f"{discount_pct:.1f}% discount yields an IRR of {irr_pct:.1f}% "
            f"vs WACC of {wacc_pct:.1f}%. Recommendation: {decision['recommendation']}."
        )

        return {**decision, "summary": summary}
    except Exception as exc:
        logger.exception("run_irr_simulation failed")
        return {"error": str(exc)}


@tool
def get_macro_indicators() -> dict:
    """Fetch the latest macroeconomic indicators from the FRED API: Fed Funds
    Rate, CPI Inflation, 10-Year Treasury Yield, and the firm's WACC from
    settings. Use this when the user asks about interest rates, inflation,
    or the cost of capital.
    """
    try:
        from scf_agent.pipeline.ingest import FREDFetcher

        indicators = FREDFetcher().get_indicators()
        summary = (
            f"Fed Funds Rate: {indicators['fed_funds_rate']:.2f}%, "
            f"CPI: {indicators['cpi_inflation']:.2f}, "
            f"10Y Treasury: {indicators['treasury_10y']:.2f}%, "
            f"WACC: {indicators['wacc'] * 100:.1f}%."
        )
        return {**indicators, "summary": summary}
    except Exception as exc:
        logger.exception("get_macro_indicators failed")
        return {"error": str(exc)}


@tool
def run_whatif_simulation(invoice_value: float, credit_score: float = 75.0) -> dict:
    """Run a full grid of early payment scenarios across multiple discount
    rates (0.5% to 3.0%) and payment windows (5 to 30 days). Returns the top
    5 most profitable opportunities ranked by IRR, plus a summary.
    Use this when the user asks for the best payment terms or wants to
    compare multiple scenarios at once.
    """
    try:
        from scf_agent.optimization.irr_engine import WhatIfSimulator

        simulator = WhatIfSimulator()
        top = simulator.top_opportunities(
            invoice_value=invoice_value,
            credit_score=credit_score,
            n=5,
        )

        if top.empty:
            return {
                "invoice_value": invoice_value,
                "credit_score": credit_score,
                "top_opportunities": [],
                "summary": "No profitable scenarios found at this credit score.",
            }

        opportunities = top.to_dict(orient="records")
        best = opportunities[0]
        summary = (
            f"Top opportunity: {int(best['days_early'])}d early at "
            f"{best['discount_pct']}% discount yields IRR of "
            f"{float(best['irr']):.1%}."
        )

        return {
            "invoice_value": invoice_value,
            "credit_score": credit_score,
            "top_opportunities": opportunities,
            "summary": summary,
        }
    except Exception as exc:
        logger.exception("run_whatif_simulation failed")
        return {"error": str(exc)}
