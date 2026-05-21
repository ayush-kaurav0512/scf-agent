"""Pydantic v2 schemas shared across the FastAPI routers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SupplierScore(BaseModel):
    supplier_name: str
    credit_score: float = Field(..., ge=0.0, le=100.0)
    distress_prob: float = Field(..., ge=0.0, le=1.0)
    risk_label: Literal["low_risk", "watch", "high_risk"]
    summary: str


class SupplierScoreResponse(BaseModel):
    data: SupplierScore


class AtRiskResponse(BaseModel):
    at_risk_count: int
    threshold: int
    suppliers: list[SupplierScore]
    summary: str


class IRRRequest(BaseModel):
    days_early: int = Field(..., gt=0, le=365)
    discount_pct: float = Field(..., gt=0.0, le=10.0)
    invoice_value: float = Field(..., gt=0.0)
    credit_score: float = Field(default=75.0, ge=0.0, le=100.0)


class IRRResponse(BaseModel):
    irr: float
    npv: float
    discount_captured: float
    wacc: float
    profitable: bool
    credit_ok: bool
    recommendation: Literal["PAY_EARLY", "HOLD", "FLAG"]
    summary: str


class WhatIfRequest(BaseModel):
    invoice_value: float = Field(..., gt=0.0)
    credit_score: float = Field(default=75.0, ge=0.0, le=100.0)
    days_range: list[int] | None = None
    discount_range: list[float] | None = None


class ScenarioRow(BaseModel):
    days_early: int
    discount_pct: float
    irr: float
    npv: float
    discount_captured: float
    profitable: bool
    recommendation: str


class WhatIfResponse(BaseModel):
    total_scenarios: int
    scenarios: list[ScenarioRow]
    top_opportunities: list[ScenarioRow]
    summary: dict


class MacroIndicatorsResponse(BaseModel):
    fed_funds_rate: float
    cpi_inflation: float
    treasury_10y: float
    wacc: float
    summary: str


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class AgentResponse(BaseModel):
    question: str
    answer: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    model_loaded: bool
