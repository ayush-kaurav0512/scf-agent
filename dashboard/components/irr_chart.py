"""IRR optimizer component for the SCF control tower."""

from __future__ import annotations

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

RECOMMENDATION_COLORS: dict[str, str] = {
    "PAY_EARLY": "#2E7D32",
    "HOLD":      "#ED6C02",
    "FLAG":      "#C62828",
}


def _fetch_irr(api_base: str, payload: dict) -> dict:
    """POST a single IRR scenario; returns the decoded JSON or an error dict."""
    try:
        resp = httpx.post(f"{api_base}/api/v1/decisions/irr", json=payload, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        return {"error": str(exc)}


def _fetch_whatif(api_base: str, payload: dict) -> dict:
    """POST a what-if grid; returns the decoded JSON or an error dict."""
    try:
        resp = httpx.post(f"{api_base}/api/v1/decisions/whatif", json=payload, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        return {"error": str(exc)}


def _render_recommendation_badge(recommendation: str, summary: str) -> None:
    text = f"**{recommendation}** — {summary}"
    if recommendation == "PAY_EARLY":
        st.success(text)
    elif recommendation == "HOLD":
        st.warning(text)
    elif recommendation == "FLAG":
        st.error(text)
    else:
        st.info(text)


def _build_scatter(scenarios: list[dict], wacc: float) -> "px.scatter":
    df = pd.DataFrame(scenarios)
    if df.empty:
        return None
    df = df.copy()
    df["irr_pct"] = df["irr"] * 100.0

    fig = px.scatter(
        df,
        x="days_early",
        y="irr_pct",
        color="recommendation",
        size="discount_pct",
        color_discrete_map=RECOMMENDATION_COLORS,
        title="Scenario grid — IRR by payment window",
        labels={
            "days_early": "Days early",
            "irr_pct": "Annualised IRR (%)",
            "recommendation": "Recommendation",
            "discount_pct": "Discount %",
        },
        hover_data={"discount_pct": ":.2f", "irr_pct": ":.2f", "recommendation": True},
    )
    fig.add_hline(
        y=wacc * 100.0,
        line_dash="dash",
        annotation_text=f"WACC {wacc * 100:.1f}%",
        annotation_position="top right",
    )
    fig.update_layout(xaxis_title="Days early", yaxis_title="Annualised IRR (%)")
    return fig


def render_irr_chart(api_base: str) -> None:
    """Render the IRR optimizer section."""
    st.subheader("IRR optimizer")

    invoice_value = st.number_input(
        "Invoice value (USD)",
        min_value=1_000,
        max_value=10_000_000,
        value=120_000,
        step=1_000,
        format="%d",
    )
    credit_score = st.slider("Supplier credit score", 0, 100, 75)
    days_early = st.slider("Days early", 5, 30, 10)
    discount_pct = st.slider("Discount %", 0.5, 4.0, 2.0, step=0.1)
    submitted = st.button("Calculate", type="primary")

    if not submitted:
        st.caption("Adjust the inputs and click Calculate to see the IRR decision.")
        return

    irr_payload = {
        "invoice_value": float(invoice_value),
        "credit_score": float(credit_score),
        "days_early": int(days_early),
        "discount_pct": float(discount_pct),
    }
    irr = _fetch_irr(api_base, irr_payload)
    if "error" in irr:
        st.error(f"IRR call failed: {irr['error']}")
        return

    m_irr, m_npv, m_disc = st.columns(3)
    m_irr.metric("Annualised IRR", f"{irr['irr'] * 100:.1f}%")
    m_npv.metric("NPV", f"${irr['npv']:,.0f}")
    m_disc.metric("Discount captured", f"${irr['discount_captured']:,.0f}")

    _render_recommendation_badge(irr["recommendation"], irr.get("summary", ""))

    whatif_payload = {
        "invoice_value": float(invoice_value),
        "credit_score": float(credit_score),
    }
    whatif = _fetch_whatif(api_base, whatif_payload)
    if "error" in whatif:
        st.error(f"What-if call failed: {whatif['error']}")
        return

    fig = _build_scatter(whatif.get("scenarios", []), wacc=float(irr["wacc"]))
    if fig is None:
        st.info("No scenarios returned for the chart.")
        return
    st.plotly_chart(fig, width="stretch")
