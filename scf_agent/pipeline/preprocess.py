"""Preprocessing stage.

Cleans the raw DataCo frame in a chain of small, idempotent steps and
persists the result to ``data/processed/dataco_clean.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from scf_agent.config import settings

logger = logging.getLogger(__name__)


class Preprocessor:
    """Fluent preprocessor for the validated DataCo frame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df.copy()

    def drop_duplicates(self) -> "Preprocessor":
        """Drop rows with duplicate ``order_id`` values."""
        before = len(self.df)
        if "order_id" in self.df.columns:
            self.df = self.df.drop_duplicates(subset=["order_id"])
        else:
            self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        logger.info("drop_duplicates: removed %s rows (kept %s)", removed, len(self.df))
        return self

    def parse_dates(self) -> "Preprocessor":
        """Coerce ``order_date`` and ``ship_date`` to datetime."""
        for col in ("order_date", "ship_date"):
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                null_count = int(self.df[col].isna().sum())
                logger.info("parse_dates: %s nulls in %s", null_count, col)
        return self

    def clean_strings(self) -> "Preprocessor":
        """Strip and lowercase all string-typed columns."""
        string_cols = self.df.select_dtypes(include=["object", "string"]).columns
        for col in string_cols:
            series = self.df[col].astype("string")
            self.df[col] = series.str.strip().str.lower()
        logger.info("clean_strings: normalized %s columns", len(string_cols))
        return self

    def fill_numerics(self) -> "Preprocessor":
        """Fill numeric NaN values with the column median."""
        numeric_cols = self.df.select_dtypes(include="number").columns
        for col in numeric_cols:
            median = self.df[col].median()
            if pd.isna(median):
                median = 0.0
            self.df[col] = self.df[col].fillna(median)
        logger.info("fill_numerics: filled %s numeric columns", len(numeric_cols))
        return self

    def add_lead_time(self) -> "Preprocessor":
        """Compute ``lead_time_days`` from order/ship dates, clamping negatives to 0."""
        if "order_date" in self.df.columns and "ship_date" in self.df.columns:
            delta = (self.df["ship_date"] - self.df["order_date"]).dt.days
            self.df["lead_time_days"] = delta.fillna(0).clip(lower=0).astype(int)
        else:
            self.df["lead_time_days"] = 0
        logger.info("add_lead_time: computed lead_time_days")
        return self

    def run(self) -> pd.DataFrame:
        """Execute the full preprocessing chain and persist the result."""
        (
            self.drop_duplicates()
                .parse_dates()
                .clean_strings()
                .fill_numerics()
                .add_lead_time()
        )

        settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path: Path = settings.DATA_PROCESSED_DIR / "dataco_clean.parquet"
        self.df.to_parquet(out_path, index=False)
        logger.info("preprocess complete: shape=%s saved to %s", self.df.shape, out_path)
        return self.df


def main() -> None:
    logging.basicConfig(level=settings.LOG_LEVEL)
    raw_path = settings.DATA_RAW_DIR / "dataco_smart_supply_chain.csv"
    df = pd.read_csv(raw_path)
    Preprocessor(df).run()


if __name__ == "__main__":
    main()
