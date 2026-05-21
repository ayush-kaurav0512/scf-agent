"""Supplier endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from scf_agent.agent.tools import get_at_risk_suppliers, get_supplier_risk_score
from scf_agent.api.schemas import AtRiskResponse, SupplierScore, SupplierScoreResponse

router = APIRouter()


@router.get("/{supplier_name}", response_model=SupplierScoreResponse)
async def get_supplier(supplier_name: str) -> SupplierScoreResponse:
    """Get credit score and risk label for a single supplier by name.

    Returns 404 if the supplier is not found in the feature matrix.
    """
    result = get_supplier_risk_score.invoke({"supplier_name": supplier_name})
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return SupplierScoreResponse(data=SupplierScore(**result))


@router.get("/", response_model=AtRiskResponse)
async def get_at_risk(
    threshold: int = Query(
        default=50,
        ge=0,
        le=100,
        description="Credit score threshold. Suppliers below this are returned.",
    ),
) -> AtRiskResponse:
    """Return all suppliers below the given credit score threshold, worst first."""
    result = get_at_risk_suppliers.invoke({"threshold": threshold})
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    suppliers_parsed = [SupplierScore(**s) for s in result["suppliers"]]
    return AtRiskResponse(
        at_risk_count=result["at_risk_count"],
        threshold=result["threshold"],
        suppliers=suppliers_parsed,
        summary=result["summary"],
    )
