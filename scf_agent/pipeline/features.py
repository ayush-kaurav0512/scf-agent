"""Feature engineering stage.

Takes the cleaned DataCo frame and produces a supplier-level feature matrix
suitable for the :class:`scf_agent.models.risk_scorer.RiskScorer`.

The output is persisted as ``data/processed/features.parquet`` and the
canonical column order is exposed as :data:`FEATURE_COLUMNS`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scf_agent.config import settings

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = (
    "lead_time_variance",
    "return_rate",
    "late_delivery_rate",
    "avg_payment_delay",
    "order_volume",
    "avg_order_value",
)


class FeatureEngineer:
    """Build a supplier-level feature matrix from a cleaned DataCo frame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df.copy()

    def _lead_time_variance(self) -> pd.Series:
        """Std of ``lead_time_days`` per supplier (0 for single-order suppliers)."""
        series = (
            self.df.groupby("supplier_name")["lead_time_days"]
            .std()
            .fillna(0.0)
        )
        series.name = "lead_time_variance"
        return series

    def _return_rate(self) -> pd.Series:
        """Share of orders whose ``delivery_status`` contains 'returned'."""
        status = self.df["delivery_status"].astype("string").fillna("")
        is_returned = status.str.contains("returned", case=False, regex=False)
        series = is_returned.groupby(self.df["supplier_name"]).mean()
        series.name = "return_rate"
        return series.astype(float)

    def _late_delivery_rate(self) -> pd.Series:
        """Mean of the 0/1 ``late_delivery_risk`` flag per supplier."""
        series = self.df.groupby("supplier_name")["late_delivery_risk"].mean()
        series.name = "late_delivery_rate"
        return series.astype(float)

    def _avg_payment_delay(self) -> pd.Series:
        """Mean positive excess of lead time over each supplier's median lead time."""
        median = self.df.groupby("supplier_name")["lead_time_days"].transform("median")
        excess = (self.df["lead_time_days"] - median).clip(lower=0)
        series = excess.groupby(self.df["supplier_name"]).mean()
        series.name = "avg_payment_delay"
        return series.astype(float)

    def _order_volume(self) -> pd.Series:
        """Number of orders per supplier."""
        series = self.df.groupby("supplier_name").size()
        series.name = "order_volume"
        return series.astype(float)

    def _avg_order_value(self) -> pd.Series:
        """Mean of ``order_item_total`` per supplier."""
        series = self.df.groupby("supplier_name")["order_item_total"].mean()
        series.name = "avg_order_value"
        return series.astype(float)

    def build(self) -> pd.DataFrame:
        """Assemble the feature matrix and persist it to parquet."""
        parts = [
            self._lead_time_variance(),
            self._return_rate(),
            self._late_delivery_rate(),
            self._avg_payment_delay(),
            self._order_volume(),
            self._avg_order_value(),
        ]
        features = pd.concat(parts, axis=1)
        features = features.reset_index()

        float_cols = features.select_dtypes(include="floating").columns
        features[float_cols] = features[float_cols].round(4)

        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path: Path = settings.DATA_PROCESSED_DIR / "features.parquet"
        features.to_parquet(out_path, index=False)

        logger.info(
            "built feature matrix: shape=%s columns=%s saved to %s",
            features.shape,
            list(features.columns),
            out_path,
        )
        return features

    @staticmethod
    def load_or_build(raw_parquet: Path) -> pd.DataFrame:
        """Return the cached feature matrix if fresh; otherwise rebuild it.

        The cache is considered fresh when ``features.parquet`` exists and
        its mtime is greater than ``raw_parquet``'s mtime.
        """
        raw_parquet = Path(raw_parquet)
        cache_path: Path = settings.DATA_PROCESSED_DIR / "features.parquet"

        if cache_path.exists() and raw_parquet.exists():
            if cache_path.stat().st_mtime > raw_parquet.stat().st_mtime:
                logger.info("feature cache hit: loading %s", cache_path)
                return pd.read_parquet(cache_path)

        if not raw_parquet.exists():
            raise FileNotFoundError(f"raw parquet not found: {raw_parquet}")

        logger.info("feature cache miss: rebuilding from %s", raw_parquet)
        df = pd.read_parquet(raw_parquet)
        return FeatureEngineer(df).build()


def main() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL)
    raw_parquet = settings.DATA_PROCESSED_DIR / "dataco_clean.parquet"
    FeatureEngineer.load_or_build(raw_parquet)
    logger.info("feature stage complete.")


if __name__ == "__main__":
    main()
