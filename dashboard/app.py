"""Streamlit entrypoint for the SCF Agent control tower."""

from __future__ import annotations

import os

import streamlit as st

from dashboard.components.irr_chart import render_irr_chart
from dashboard.components.nl_panel import render_nl_panel
from dashboard.components.risk_table import render_risk_table

DEFAULT_API_BASE: str = os.environ.get("API_BASE_URL", "http://localhost:8000")


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("SCF Control Tower")
        st.caption("Supply Chain Finance AI Agent")
        st.divider()

        api_base = st.text_input(
            "API base URL",
            value=st.session_state.get("api_base", DEFAULT_API_BASE),
            help="Root URL of the FastAPI service.",
        )
        st.session_state["api_base"] = api_base

        threshold = st.slider(
            "Risk threshold",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("threshold", 50)),
            help="Suppliers with a credit score below this are flagged at-risk.",
        )
        st.session_state["threshold"] = threshold

        if st.button("Refresh data"):
            st.cache_data.clear()
            st.toast("Cache cleared", icon=None)

        st.divider()
        st.caption("Powered by LangChain + Claude")


def main() -> None:
    st.set_page_config(
        page_title="SCF Control Tower",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _render_sidebar()

    api_base: str = st.session_state.get("api_base", DEFAULT_API_BASE)
    threshold: int = int(st.session_state.get("threshold", 50))

    col1, col2 = st.columns([2, 1])

    with col1:
        try:
            render_risk_table(api_base=api_base, threshold=threshold)
        except Exception as exc:
            st.error(f"Risk table failed to render: {exc}")

    with col2:
        try:
            render_irr_chart(api_base=api_base)
        except Exception as exc:
            st.error(f"IRR chart failed to render: {exc}")

    st.divider()

    try:
        render_nl_panel(api_base=api_base)
    except Exception as exc:
        st.error(f"NL panel failed to render: {exc}")


if __name__ == "__main__":
    main()
