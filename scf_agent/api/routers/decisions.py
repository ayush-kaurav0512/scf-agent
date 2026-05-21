"""Decisioning endpoints — single-invoice IRR, what-if grid, and macro indicators."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from scf_agent.agent.tools import (
    get_macro_indicators,
    run_irr_simulation,
    run_whatif_simulation,
)
from scf_agent.api.schemas import (
    IRRRequest,
    IRRResponse,
    MacroIndicatorsResponse,
    ScenarioRow,
    WhatIfRequest,
    WhatIfResponse,
)

router = APIRouter()


@router.post("/irr", response_model=IRRResponse)
async def calculate_irr(req: IRRRequest) -> IRRResponse:
    """Calculate IRR, NPV, and a PAY_EARLY / HOLD / FLAG recommendation."""
    result = run_irr_simulation.invoke(req.model_dump())
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return IRRResponse(**result)


@router.post("/whatif", response_model=WhatIfResponse)
async def whatif(req: WhatIfRequest) -> WhatIfResponse:
    """Run a full scenario grid across discount rates and payment windows."""
    result = run_whatif_simulation.invoke(req.model_dump())
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    from scf_agent.optimization.irr_engine import WhatIfSimulator

    simulator = WhatIfSimulator()
    full_df = simulator.run(
        invoice_value=req.invoice_value,
        credit_score=req.credit_score,
        days_range=req.days_range,
        discount_range=req.discount_range,
    )
    summary_dict = simulator.summary(req.invoice_value, req.credit_score)
    top = result.get("top_opportunities", [])

    scenarios = [ScenarioRow(**row) for row in full_df.to_dict(orient="records")]
    top_rows = [ScenarioRow(**row) for row in top]

    return WhatIfResponse(
        total_scenarios=len(scenarios),
        scenarios=scenarios,
        top_opportunities=top_rows,
        summary=summary_dict,
    )


@router.get("/macro", response_model=MacroIndicatorsResponse)
async def macro_indicators() -> MacroIndicatorsResponse:
    """Fetch the latest macroeconomic indicators from FRED."""
    result = get_macro_indicators.invoke({})
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return MacroIndicatorsResponse(**result)
