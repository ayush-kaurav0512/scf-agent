"""Natural-language chat panel for the SCF control tower."""

from __future__ import annotations

import httpx
import streamlit as st

SUGGESTED_QUESTIONS: list[str] = [
    "Which suppliers are highest risk?",
    "Should I pay early on a $120k invoice at 2% discount for 10 days?",
    "What are current interest rates?",
]


def _ask_agent(api_base: str, question: str) -> dict:
    """POST a question to the SCF agent; returns the decoded JSON or an error dict."""
    try:
        resp = httpx.post(
            f"{api_base}/api/v1/agent/ask",
            json={"question": question},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        return {"error": str(exc)}


def _append_turn(messages: list, role: str, content: str) -> list:
    """Pure helper: append a chat turn to the messages list and return it."""
    messages.append({"role": role, "content": content})
    return messages


def _render_history(messages: list) -> None:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _render_suggestions() -> None:
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, question in zip(cols, SUGGESTED_QUESTIONS):
        if col.button(question, key=f"suggest::{question}"):
            st.session_state["pending_question"] = question
            st.rerun()


def render_nl_panel(api_base: str) -> None:
    """Render the chat-style natural-language query panel."""
    st.subheader("Ask the agent")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if not st.session_state["messages"]:
        st.caption("Try one of these to get started:")
        _render_suggestions()

    _render_history(st.session_state["messages"])

    question = st.chat_input("Ask about suppliers, risk, or payment decisions...")
    if not question and "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if not question:
        return

    _append_turn(st.session_state["messages"], "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = _ask_agent(api_base, question)
        if "error" in payload:
            answer = f"Agent call failed: {payload['error']}"
            st.error(answer)
        else:
            answer = payload.get("answer", "")
            st.markdown(answer)

    _append_turn(st.session_state["messages"], "assistant", answer)
