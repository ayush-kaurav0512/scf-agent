"""Natural-language SCF agent.

Builds a tool-calling ReAct agent on top of ``ChatGroq`` (Llama 3.3 70B)
using LangGraph's prebuilt ``create_react_agent``. The public surface is a
single async :func:`ask` function consumed by the FastAPI ``/agent/ask`` route.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from scf_agent.agent.memory import get_memory
from scf_agent.agent.tools import (
    get_at_risk_suppliers,
    get_macro_indicators,
    get_supplier_risk_score,
    run_irr_simulation,
    run_whatif_simulation,
)
from scf_agent.config import settings

logger = logging.getLogger(__name__)

TOOLS = [
    get_supplier_risk_score,
    get_at_risk_suppliers,
    run_irr_simulation,
    get_macro_indicators,
    run_whatif_simulation,
]

SYSTEM_PROMPT = """\
You are an expert Supply Chain Finance analyst AI assistant.
You have access to real-time supplier risk scores, macroeconomic indicators,
and an IRR optimization engine.

When answering:
- Always ground your answer in tool output, not assumptions.
- Be concise but precise. Quote scores and percentages exactly.
- If a supplier is high risk, say so clearly and explain why.
- If asked for a recommendation, give one — do not hedge excessively.
- Format numbers clearly: credit scores as X/100, IRR as X.X%, NPV as $X,XXX.

You have memory of the last 10 exchanges in this session.
"""

_llm: ChatGroq | None = None
_memory = get_memory()


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=settings.GROQ_API_KEY,
        )
    return _llm


def _run_agent(question: str) -> str:
    llm = _get_llm()
    agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)

    history = _memory.get_messages()
    messages = history + [HumanMessage(content=question)]

    result = agent.invoke({"messages": messages})

    all_messages = result["messages"]
    _memory.add_messages([HumanMessage(content=question), all_messages[-1]])

    return all_messages[-1].content


async def ask(question: str) -> str:
    """Ask the SCF agent a question. Runs the sync agent in a thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_agent, question)
