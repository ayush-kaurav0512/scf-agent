"""Supplier credit-risk scorer.

A :class:`SupplierRiskScorer` wraps a soft-voting ensemble of
``LogisticRegression`` + ``XGBClassifier``. Because the DataCo dataset does
not ship with a ground-truth distress label, one is engineered from the
behavioural features (high late-delivery rate AND return rate AND lead-time
variance). The output of :meth:`predict_scores` is a 0-100 credit score per
supplier with a categorical ``risk_label``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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

DISTRESS_LATE_DELIVERY_RATE: float = 0.4
DISTRESS_RETURN_RATE: float = 0.03
DISTRESS_LEAD_TIME_VARIANCE: float = 3.0


def _default_artifact_path() -> Path:
    return settings.MODEL_ARTIFACTS_DIR / f"risk_scorer_v{settings.MODEL_VERSION}.joblib"


class SupplierRiskScorer:
    """LR + XGBoost soft-voting ensemble producing 0-100 supplier credit scores."""

    def __init__(self) -> None:
        self.feature_cols: list[str] = list(FEATURE_COLUMNS)
        self.scaler: StandardScaler = StandardScaler()
        self.lr: LogisticRegression = LogisticRegression(max_iter=1000, random_state=42)
        self.xgb: XGBClassifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        self.ensemble: VotingClassifier = VotingClassifier(
            estimators=[("lr", self.lr), ("xgb", self.xgb)],
            voting="soft",
        )
        self._is_trained: bool = False

    def _make_labels(self, df: pd.DataFrame) -> pd.Series:
        """Synthesize a binary distress label from supplier features."""
        distress = (
            (df["late_delivery_rate"] > DISTRESS_LATE_DELIVERY_RATE)
            & (df["return_rate"] > DISTRESS_RETURN_RATE)
            & (df["lead_time_variance"] > DISTRESS_LEAD_TIME_VARIANCE)
        ).astype(int)
        distress.name = "distress"

        counts = distress.value_counts().to_dict()
        logger.info(
            "distress class balance: distress=0 -> %s, distress=1 -> %s",
            int(counts.get(0, 0)),
            int(counts.get(1, 0)),
        )
        return distress

    def _guard_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Fit the ensemble on ``df`` and return held-out metrics."""
        y = self._make_labels(df)

        X = df[self.feature_cols]
        assert not X.isna().any().any(), "training features contain nulls."

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.ensemble.fit(X_train_s, y_train)
        self._is_trained = True

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            logger.warning(
                "train/test split has a single class (train=%s, test=%s); "
                "skipping evaluation metrics.",
                sorted(y_train.unique().tolist()),
                sorted(y_test.unique().tolist()),
            )
            return {}

        proba_test = self.ensemble.predict_proba(X_test_s)[:, 1]
        pred_test = self.ensemble.predict(X_test_s)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred_test)),
            "roc_auc": float(roc_auc_score(y_test, proba_test)),
            "f1": float(f1_score(y_test, pred_test, zero_division=0)),
        }
        logger.info(
            "trained SupplierRiskScorer: accuracy=%.4f roc_auc=%.4f f1=%.4f",
            metrics["accuracy"],
            metrics["roc_auc"],
            metrics["f1"],
        )
        return metrics

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Produce a per-supplier credit score and risk label."""
        self._guard_trained()

        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        distress_prob = self.ensemble.predict_proba(X_scaled)[:, 1]
        credit_score = np.round((1.0 - distress_prob) * 100.0, 1)

        safe = float(settings.RISK_THRESHOLD_SAFE)
        watch = float(settings.RISK_THRESHOLD_WATCH)
        risk_label = np.where(
            credit_score >= safe,
            "low_risk",
            np.where(credit_score >= watch, "watch", "high_risk"),
        )

        return pd.DataFrame(
            {
                "supplier_name": df["supplier_name"].to_numpy(),
                "credit_score": credit_score,
                "distress_prob": distress_prob,
                "risk_label": risk_label,
            }
        )

    def save(self, path: Path | None = None) -> Path:
        """Persist the trained scorer to disk."""
        self._guard_trained()
        target = Path(path) if path is not None else _default_artifact_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        logger.info("saved SupplierRiskScorer to %s", target)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "SupplierRiskScorer":
        """Restore a trained scorer from disk."""
        source = Path(path) if path is not None else _default_artifact_path()
        if not source.exists():
            raise FileNotFoundError(
                f"risk scorer artifact not found at {source}. "
                f"Train one first with SupplierRiskScorer.train_and_save()."
            )

        scorer = joblib.load(source)
        if not isinstance(scorer, cls):
            raise TypeError(f"loaded object at {source} is not a {cls.__name__}.")
        if not scorer._is_trained:
            raise RuntimeError(f"loaded scorer at {source} is not in a trained state.")

        logger.info("loaded SupplierRiskScorer from %s", source)
        return scorer

    @classmethod
    def train_and_save(cls, features_path: Path) -> "SupplierRiskScorer":
        """Convenience: read features parquet, train, save, return the scorer."""
        df = pd.read_parquet(Path(features_path))
        scorer = cls()
        scorer.train(df)
        scorer.save()
        return scorer


# ---------------------------------------------------------------------------
# Backward-compat helper (consumed by scf_agent.api.routers.suppliers).
# ---------------------------------------------------------------------------

@dataclass
class RiskBand:
    label: str
    color: str


def classify_band(score: float) -> RiskBand:
    """Map a 0-100 credit score to the legacy SAFE / WATCH / AT_RISK band."""
    if score >= settings.RISK_THRESHOLD_SAFE:
        return RiskBand(label="SAFE", color="green")
    if score >= settings.RISK_THRESHOLD_WATCH:
        return RiskBand(label="WATCH", color="amber")
    return RiskBand(label="AT_RISK", color="red")
