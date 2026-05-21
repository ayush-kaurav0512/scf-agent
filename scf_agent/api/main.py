"""FastAPI application entrypoint for the SCF agent service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scf_agent.api.routers import agent, decisions, suppliers
from scf_agent.api.schemas import HealthResponse
from scf_agent.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SCF Agent API starting up...")
    yield
    logger.info("SCF Agent API shutting down.")


app = FastAPI(
    title="SCF Agent API",
    description="Supply Chain Finance AI Agent — risk scoring, IRR optimization, NL queries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(suppliers.router, prefix="/api/v1/suppliers", tags=["Suppliers"])
app.include_router(decisions.router, prefix="/api/v1/decisions", tags=["Decisions"])
app.include_router(agent.router,     prefix="/api/v1/agent",     tags=["Agent"])


@app.get("/health", response_model=HealthResponse)
async def health():
    model_path = Path("scf_agent/models/artifacts")
    model_loaded = any(model_path.glob("*.joblib")) if model_path.exists() else False
    return HealthResponse(
        status="ok",
        version=settings.MODEL_VERSION,
        model_loaded=model_loaded,
    )
