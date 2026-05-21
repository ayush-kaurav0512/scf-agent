"""Ingestion stage.

Two ingesters live here:

* :class:`DataCoIngester` — reads the DataCo Smart Supply Chain CSV from
  ``data/raw/`` and validates its schema.
* :class:`FREDFetcher` — pulls the latest values for a handful of FRED
  macroeconomic series and packages them as a flat ``dict``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pandas as pd

from scf_agent.config import settings

logger = logging.getLogger(__name__)

FRED_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"
HTTP_TIMEOUT_SECONDS: float = 10.0

REQUIRED_DATACO_COLUMNS: tuple[str, ...] = (
    "order_id",
    "customer_id",
    "order_date",
    "ship_date",
    "delivery_status",
    "late_delivery_risk",
    "order_item_quantity",
    "order_item_total",
    "supplier_name",
    "product_category",
)

MIN_DATACO_ROWS: int = 1000


class DataCoIngester:
    """Reads and validates the DataCo Smart Supply Chain CSV."""

    def __init__(self, raw_path: Path) -> None:
        self.raw_path: Path = Path(raw_path)

    def load(self) -> pd.DataFrame:
        """Read the CSV from ``self.raw_path``.

        Raises:
            FileNotFoundError: if the path does not exist, with guidance on
                where to drop the DataCo file.
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"DataCo CSV not found at {self.raw_path}. "
                f"Download the Smart Supply Chain dataset and place it under "
                f"{settings.DATA_RAW_DIR}/ before running the pipeline."
            )

        df = pd.read_csv(self.raw_path)
        logger.info(
            "loaded DataCo CSV from %s: shape=%s columns=%s",
            self.raw_path,
            df.shape,
            list(df.columns),
        )
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verify required columns and minimum row count."""
        missing = [c for c in REQUIRED_DATACO_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataCo CSV is missing required columns: {missing}. "
                f"Present columns: {list(df.columns)}"
            )

        if len(df) < MIN_DATACO_ROWS:
            raise ValueError(
                f"DataCo CSV has only {len(df)} rows; at least {MIN_DATACO_ROWS} required."
            )

        logger.info("validated DataCo CSV: rows=%s", len(df))
        return df

    def run(self) -> pd.DataFrame:
        """Load and validate the CSV in one call."""
        return self.validate(self.load())


class FREDFetcher:
    """Fetches the latest values of selected FRED macro series."""

    INDICATOR_SERIES: dict[str, str] = {
        "fed_funds_rate": "FEDFUNDS",
        "cpi_inflation": "CPIAUCSL",
        "treasury_10y": "DGS10",
    }

    def __init__(self) -> None:
        self.api_key: str = settings.FRED_API_KEY
        self.base_url: str = FRED_BASE_URL

    def _fetch_series(self, series_id: str, limit: int = 1) -> float:
        """Return the most recent numeric observation for ``series_id``."""
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY is not configured.")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "sort_order": "desc",
            "limit": limit,
            "file_type": "json",
        }

        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(self.base_url, params=params)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"FRED request failed for {series_id}: {exc}") from exc

        observations = payload.get("observations") or []
        if not observations:
            raise RuntimeError(f"FRED returned no observations for {series_id}.")

        raw_value = observations[0].get("value")
        if raw_value is None or raw_value in {"", "."}:
            raise RuntimeError(f"FRED latest observation for {series_id} is missing.")

        try:
            return float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"FRED returned non-numeric value '{raw_value}' for {series_id}."
            ) from exc

    def get_indicators(self) -> dict[str, float]:
        """Return all configured indicators plus WACC as a flat dict."""
        indicators: dict[str, float] = {}
        for key, series_id in self.INDICATOR_SERIES.items():
            indicators[key] = self._fetch_series(series_id)
        indicators["wacc"] = float(settings.WACC)
        logger.info("fetched FRED indicators: %s", indicators)
        return indicators


def main() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL)

    dataco_path = settings.DATA_RAW_DIR / "dataco_smart_supply_chain.csv"
    try:
        DataCoIngester(dataco_path).run()
    except FileNotFoundError as exc:
        logger.warning("%s", exc)

    if settings.FRED_API_KEY:
        FREDFetcher().get_indicators()
    else:
        logger.warning("FRED_API_KEY missing; skipping FRED fetch.")

    logger.info("ingest stage complete.")


if __name__ == "__main__":
    main()
