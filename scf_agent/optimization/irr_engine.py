"""IRR / NPV engine for early-payment-discount decisioning.

Two classes drive the module:

* :class:`IRRCalculator` — single-invoice math (IRR, captured discount, NPV)
  and the structured PAY_EARLY / HOLD / FLAG decision.
* :class:`WhatIfSimulator` — runs a grid of (days_early × discount_pct)
  scenarios on a single invoice and ranks them.

A small set of free functions (:func:`annualized_irr`, :func:`simulate`,
:func:`sweep`, :func:`best_offer`) is preserved for the existing FastAPI
``/decisions/irr`` route and its response schema; new code should use the
class-based interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import pandas as pd

from scf_agent.config import settings

DAYS_PER_YEAR: int = 365

WHATIF_COLUMNS: tuple[str, ...] = (
    "days_early",
    "discount_pct",
    "irr",
    "npv",
    "discount_captured",
    "profitable",
    "recommendation",
)

DEFAULT_DAYS_RANGE: tuple[int, ...] = (5, 10, 15, 20, 30)
DEFAULT_DISCOUNT_RANGE: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


# ---------------------------------------------------------------------------
# IRRCalculator
# ---------------------------------------------------------------------------

class IRRCalculator:
    """Single-invoice early-payment math."""

    def __init__(self, wacc: float | None = None) -> None:
        self.wacc: float = wacc if wacc is not None else float(settings.WACC)

    def annualised_irr(self, discount_pct: float, days_early: int) -> float:
        """Annualised IRR of the early-payment offer.

        ``discount_pct`` is a percentage (``2`` means 2%), not a fraction.
        """
        if days_early <= 0:
            raise ValueError("days_early must be > 0")
        if discount_pct <= 0:
            raise ValueError("discount_pct must be > 0")
        return (discount_pct / 100.0) * (DAYS_PER_YEAR / days_early)

    def discount_captured(self, invoice_value: float, discount_pct: float) -> float:
        """Dollar value of the discount on a single invoice."""
        if invoice_value <= 0:
            raise ValueError("invoice_value must be > 0")
        return invoice_value * (discount_pct / 100.0)

    def npv(self, invoice_value: float, discount_pct: float, days_early: int) -> float:
        """Net present value of paying ``days_early`` days early at ``discount_pct``."""
        cash_in = self.discount_captured(invoice_value, discount_pct)
        cash_out = invoice_value * self.wacc * (days_early / DAYS_PER_YEAR)
        return cash_in - cash_out

    def decision(
        self,
        discount_pct: float,
        days_early: int,
        invoice_value: float,
        credit_score: float,
    ) -> dict:
        """Return the structured PAY_EARLY / HOLD / FLAG decision for one offer."""
        irr = self.annualised_irr(discount_pct, days_early)
        npv_value = self.npv(invoice_value, discount_pct, days_early)
        captured = self.discount_captured(invoice_value, discount_pct)

        profitable = irr > self.wacc
        credit_ok = credit_score >= settings.RISK_THRESHOLD_WATCH

        if not credit_ok:
            recommendation = "FLAG"
        elif profitable:
            recommendation = "PAY_EARLY"
        else:
            recommendation = "HOLD"

        return {
            "irr": round(float(irr), 4),
            "npv": round(float(npv_value), 4),
            "discount_captured": round(float(captured), 4),
            "wacc": self.wacc,
            "profitable": bool(profitable),
            "credit_ok": bool(credit_ok),
            "recommendation": recommendation,
        }


# ---------------------------------------------------------------------------
# WhatIfSimulator
# ---------------------------------------------------------------------------

class WhatIfSimulator:
    """Runs a grid of scenarios on a single invoice and ranks them by IRR."""

    def __init__(self, calculator: IRRCalculator | None = None) -> None:
        self.calc: IRRCalculator = calculator or IRRCalculator()

    def run(
        self,
        invoice_value: float,
        credit_score: float,
        days_range: list[int] | None = None,
        discount_range: list[float] | None = None,
    ) -> pd.DataFrame:
        """Evaluate the full days × discount grid and return a ranked DataFrame."""
        days = list(days_range) if days_range is not None else list(DEFAULT_DAYS_RANGE)
        discounts = list(discount_range) if discount_range is not None else list(DEFAULT_DISCOUNT_RANGE)

        rows: list[dict] = []
        for days_early, discount_pct in product(days, discounts):
            decision = self.calc.decision(
                discount_pct=discount_pct,
                days_early=days_early,
                invoice_value=invoice_value,
                credit_score=credit_score,
            )
            rows.append(
                {
                    "days_early": days_early,
                    "discount_pct": discount_pct,
                    "irr": decision["irr"],
                    "npv": decision["npv"],
                    "discount_captured": decision["discount_captured"],
                    "profitable": decision["profitable"],
                    "recommendation": decision["recommendation"],
                }
            )

        df = pd.DataFrame(rows, columns=list(WHATIF_COLUMNS))
        df = df.sort_values("irr", ascending=False).reset_index(drop=True)

        float_cols = df.select_dtypes(include="floating").columns
        df[float_cols] = df[float_cols].round(4)
        return df

    def top_opportunities(
        self,
        invoice_value: float,
        credit_score: float,
        n: int = 5,
    ) -> pd.DataFrame:
        """Top ``n`` PAY_EARLY scenarios by IRR (empty frame if none)."""
        scenarios = self.run(invoice_value=invoice_value, credit_score=credit_score)
        pay_early = scenarios[scenarios["recommendation"] == "PAY_EARLY"]
        if pay_early.empty:
            return pd.DataFrame(columns=list(WHATIF_COLUMNS))
        return pay_early.head(n).reset_index(drop=True)

    def summary(self, invoice_value: float, credit_score: float) -> dict:
        """Aggregate stats over the default scenario grid."""
        scenarios = self.run(invoice_value=invoice_value, credit_score=credit_score)
        total = len(scenarios)
        profitable_count = int(scenarios["profitable"].sum())

        if total == 0:
            return {
                "total_scenarios": 0,
                "profitable_count": 0,
                "best_irr": 0.0,
                "best_recommendation": "",
                "avg_npv": 0.0,
            }

        best_row = scenarios.iloc[0]
        return {
            "total_scenarios": total,
            "profitable_count": profitable_count,
            "best_irr": float(best_row["irr"]),
            "best_recommendation": str(best_row["recommendation"]),
            "avg_npv": float(scenarios["npv"].mean()),
        }


# ---------------------------------------------------------------------------
# Backward-compat free-function API (consumed by scf_agent.api.routers.decisions).
# ---------------------------------------------------------------------------

@dataclass
class IRRResult:
    """Legacy single-scenario result used by the FastAPI /decisions/irr route."""

    invoice_value: float
    days_early: int
    discount_pct: float
    discount_amount: float
    cash_paid: float
    annualized_irr: float
    wacc: float
    spread_vs_wacc: float
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


def annualized_irr(*, discount_pct: float, days_early: int) -> float:
    """Legacy IRR using a fractional discount (e.g. ``0.02`` for 2%)."""
    if days_early <= 0:
        raise ValueError("days_early must be > 0")
    if not 0 < discount_pct < 1:
        raise ValueError("discount_pct must be a fraction in (0, 1)")
    return (discount_pct / (1.0 - discount_pct)) * (DAYS_PER_YEAR / days_early)


def simulate(
    *,
    days_early: int,
    discount_pct: float,
    invoice_value: float,
    wacc: float | None = None,
) -> IRRResult:
    """Legacy per-invoice simulation used by the existing FastAPI route."""
    wacc_used = wacc if wacc is not None else settings.WACC

    discount_amount = invoice_value * discount_pct
    cash_paid = invoice_value - discount_amount
    irr = annualized_irr(discount_pct=discount_pct, days_early=days_early)
    spread = irr - wacc_used

    if spread > 0.02:
        recommendation = "ACCEPT"
    elif spread > 0:
        recommendation = "CONSIDER"
    else:
        recommendation = "REJECT"

    return IRRResult(
        invoice_value=invoice_value,
        days_early=days_early,
        discount_pct=discount_pct,
        discount_amount=discount_amount,
        cash_paid=cash_paid,
        annualized_irr=irr,
        wacc=wacc_used,
        spread_vs_wacc=spread,
        recommendation=recommendation,
    )


def sweep(
    *,
    invoice_value: float,
    days_early_grid: Iterable[int] = range(5, 61, 5),
    discount_grid: Iterable[float] = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03),
    wacc: float | None = None,
) -> list[IRRResult]:
    """Legacy grid sweep returning a list of :class:`IRRResult`."""
    results: list[IRRResult] = []
    for d in days_early_grid:
        for pct in discount_grid:
            results.append(
                simulate(
                    days_early=d,
                    discount_pct=pct,
                    invoice_value=invoice_value,
                    wacc=wacc,
                )
            )
    return results


def best_offer(results: Iterable[IRRResult]) -> IRRResult | None:
    """Legacy: return the offer with the highest IRR (or None if empty)."""
    results = list(results)
    if not results:
        return None
    return max(results, key=lambda r: r.annualized_irr)
