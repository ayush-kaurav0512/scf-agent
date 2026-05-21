"""Natural-language agent endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from scf_agent.agent.nl_agent import ask
from scf_agent.api.schemas import AgentRequest, AgentResponse

router = APIRouter()


@router.post("/ask", response_model=AgentResponse)
async def ask_agent(req: AgentRequest) -> AgentResponse:
    """Send a plain-English question to the SCF LangChain agent.

    The agent will invoke the appropriate tools and return a grounded,
    precise financial answer.

    Example questions:
    - "Which suppliers are highest risk this week?"
    - "Should I pay SteelCore early on a $120,000 invoice at 2% discount?"
    - "What are the current interest rates?"
    """
    try:
        answer = await ask(req.question)
        return AgentResponse(question=req.question, answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
