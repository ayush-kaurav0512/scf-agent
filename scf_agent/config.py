"""Centralized application configuration.

All environment variables are loaded once, validated by pydantic, and exposed
through a single ``settings`` singleton that is imported throughout the codebase.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment / .env."""

    GROQ_API_KEY: str = Field(default="", description="Groq API key.")
    FRED_API_KEY: str = Field(default="", description="FRED API key for macro data.")

    MODEL_VERSION: str = Field(default="v1", description="Risk-scorer model version tag.")
    WACC: float = Field(default=0.084, description="Weighted average cost of capital used by the IRR engine.")

    RISK_THRESHOLD_SAFE: int = Field(default=70, description="Score >= this is considered safe.")
    RISK_THRESHOLD_WATCH: int = Field(default=50, description="Score in [WATCH, SAFE) is on the watchlist.")

    LOG_LEVEL: str = Field(default="INFO", description="Root logging level.")

    DATA_RAW_DIR: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    DATA_PROCESSED_DIR: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    MODEL_ARTIFACTS_DIR: Path = Field(default=PROJECT_ROOT / "scf_agent" / "models" / "artifacts")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings: Settings = Settings()
