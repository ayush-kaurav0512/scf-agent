"""Supplier risk table component for the SCF control tower."""

from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

ROW_COLORS: dict[str, str] = {
    "high_risk": "#FCEBEB",
    "watch":     "#FAEEDA",
    "low_risk":  "#EAF3DE",
}

DISPLAY_COLUMNS: list[str] = ["supplier_name", "credit_score", "distress_prob", "risk_label"]


@st.cache_data(ttl=300)
def _fetch_at_risk(api_base: str, threshold: int) -> dict:
    """Fetch suppliers below the credit-score threshold from the SCF API."""
    try:
        resp = httpx.get(
            f"{api_base}/api/v1/suppliers/",
            params={"threshold": threshold},
            timeout=60.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def _style_rows(row: pd.Series) -> list[str]:
    color = ROW_COLORS.get(str(row.get("risk_label", "")), "")
    style = f"background-color: {color}" if color else ""
    return [style] * len(row)


def render_risk_table(api_base: str, threshold: int) -> None:
    """Render the supplier risk-score section."""
    st.subheader("Supplier risk scores")

    payload = _fetch_at_risk(api_base, threshold)
    if "error" in payload:
        st.error(f"Failed to fetch supplier data: {payload['error']}")
        return

    suppliers = payload.get("suppliers", [])
    at_risk_count = int(payload.get("at_risk_count", len(suppliers)))
    df = pd.DataFrame(suppliers)

    total_suppliers = len(df)
    avg_score = float(df["credit_score"].mean()) if not df.empty else 0.0

    col_total, col_at_risk, col_avg = st.columns(3)
    col_total.metric("Total suppliers", f"{total_suppliers}")
    col_at_risk.metric("At-risk count", f"{at_risk_count}")
    col_avg.metric("Avg credit score", f"{avg_score:.1f}")

    if df.empty:
        st.info("No suppliers returned. Adjust the threshold or check the API.")
    else:
        ordered = df[[c for c in DISPLAY_COLUMNS if c in df.columns]]
        styled = ordered.style.apply(_style_rows, axis=1).format(
            {"credit_score": "{:.1f}", "distress_prob": "{:.3f}"}
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    if at_risk_count == 0:
        st.success("All suppliers healthy.")
    else:
        st.warning(f"{at_risk_count} suppliers need attention.")
