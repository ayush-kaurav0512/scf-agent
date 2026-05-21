"""Tests for the IRRCalculator and WhatIfSimulator."""

from __future__ import annotations

import pytest

from scf_agent.optimization.irr_engine import (
    DEFAULT_DAYS_RANGE,
    DEFAULT_DISCOUNT_RANGE,
    WHATIF_COLUMNS,
    IRRCalculator,
    WhatIfSimulator,
)


# ---------------------------------------------------------------------------
# IRRCalculator
# ---------------------------------------------------------------------------

def test_annualised_irr_formula() -> None:
    calc = IRRCalculator(wacc=0.084)
    assert calc.annualised_irr(discount_pct=2.0, days_early=10) == pytest.approx(0.73)


def test_annualised_irr_raises_on_zero_days() -> None:
    calc = IRRCalculator(wacc=0.084)
    with pytest.raises(ValueError, match="days_early"):
        calc.annualised_irr(discount_pct=2.0, days_early=0)


def test_npv_positive_when_profitable() -> None:
    calc = IRRCalculator(wacc=0.084)
    npv = calc.npv(invoice_value=500_000.0, discount_pct=3.0, days_early=5)
    assert npv > 0


def test_npv_negative_when_unprofitable() -> None:
    calc = IRRCalculator(wacc=0.084)
    npv = calc.npv(invoice_value=100_000.0, discount_pct=0.1, days_early=90)
    assert npv < 0


def test_decision_flag_on_low_credit() -> None:
    calc = IRRCalculator(wacc=0.084)
    out = calc.decision(
        discount_pct=2.0,
        days_early=10,
        invoice_value=100_000.0,
        credit_score=20,
    )
    assert out["recommendation"] == "FLAG"
    assert out["credit_ok"] is False
    assert out["profitable"] is True


def test_decision_pay_early() -> None:
    calc = IRRCalculator(wacc=0.084)
    out = calc.decision(
        discount_pct=2.0,
        days_early=10,
        invoice_value=100_000.0,
        credit_score=80,
    )
    assert out["recommendation"] == "PAY_EARLY"
    assert out["profitable"] is True
    assert out["credit_ok"] is True


def test_decision_hold() -> None:
    calc = IRRCalculator(wacc=0.084)
    out = calc.decision(
        discount_pct=0.5,
        days_early=60,
        invoice_value=100_000.0,
        credit_score=80,
    )
    assert out["recommendation"] == "HOLD"
    assert out["profitable"] is False
    assert out["credit_ok"] is True


# ---------------------------------------------------------------------------
# WhatIfSimulator
# ---------------------------------------------------------------------------

def test_whatif_run_shape() -> None:
    sim = WhatIfSimulator(IRRCalculator(wacc=0.084))
    df = sim.run(invoice_value=100_000.0, credit_score=80)

    assert len(df) == len(DEFAULT_DAYS_RANGE) * len(DEFAULT_DISCOUNT_RANGE) == 30
    assert list(df.columns) == list(WHATIF_COLUMNS)


def test_whatif_sorted_by_irr() -> None:
    sim = WhatIfSimulator(IRRCalculator(wacc=0.084))
    df = sim.run(invoice_value=100_000.0, credit_score=80)

    irrs = df["irr"].to_list()
    assert all(earlier >= later for earlier, later in zip(irrs, irrs[1:]))


def test_top_opportunities_all_pay_early() -> None:
    sim = WhatIfSimulator(IRRCalculator(wacc=0.084))
    top = sim.top_opportunities(invoice_value=100_000.0, credit_score=80, n=5)

    assert not top.empty
    assert (top["recommendation"] == "PAY_EARLY").all()
    assert len(top) <= 5


def test_top_opportunities_returns_empty_when_no_pay_early() -> None:
    sim = WhatIfSimulator(IRRCalculator(wacc=0.084))
    empty = sim.top_opportunities(invoice_value=100_000.0, credit_score=20, n=5)

    assert empty.empty
    assert list(empty.columns) == list(WHATIF_COLUMNS)


def test_summary_keys() -> None:
    sim = WhatIfSimulator(IRRCalculator(wacc=0.084))
    summary = sim.summary(invoice_value=100_000.0, credit_score=80)

    assert set(summary.keys()) == {
        "total_scenarios",
        "profitable_count",
        "best_irr",
        "best_recommendation",
        "avg_npv",
    }
    assert summary["total_scenarios"] == 30
    assert summary["profitable_count"] >= 1
    assert summary["best_irr"] > 0
