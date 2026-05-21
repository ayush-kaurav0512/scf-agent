"""Tests for the SupplierRiskScorer ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scf_agent.models.risk_scorer import SupplierRiskScorer


@pytest.fixture
def feature_matrix() -> pd.DataFrame:
    """Synthetic supplier feature matrix with both distress classes represented."""
    rng = np.random.default_rng(42)

    n_distress = 25
    n_healthy = 35

    distress = pd.DataFrame(
        {
            "supplier_name": [f"distress_{i:02d}" for i in range(n_distress)],
            "lead_time_variance": rng.uniform(3.5, 8.0, n_distress),
            "return_rate":        rng.uniform(0.05, 0.20, n_distress),
            "late_delivery_rate": rng.uniform(0.50, 0.95, n_distress),
            "avg_payment_delay":  rng.uniform(2.0, 12.0, n_distress),
            "order_volume":       rng.uniform(20, 200, n_distress),
            "avg_order_value":    rng.uniform(50.0, 500.0, n_distress),
        }
    )
    healthy = pd.DataFrame(
        {
            "supplier_name": [f"healthy_{i:02d}" for i in range(n_healthy)],
            "lead_time_variance": rng.uniform(0.0, 2.5, n_healthy),
            "return_rate":        rng.uniform(0.0, 0.025, n_healthy),
            "late_delivery_rate": rng.uniform(0.0, 0.35, n_healthy),
            "avg_payment_delay":  rng.uniform(0.0, 4.0, n_healthy),
            "order_volume":       rng.uniform(20, 200, n_healthy),
            "avg_order_value":    rng.uniform(50.0, 500.0, n_healthy),
        }
    )

    return pd.concat([distress, healthy], ignore_index=True)


def test_make_labels_is_binary(feature_matrix: pd.DataFrame) -> None:
    scorer = SupplierRiskScorer()
    labels = scorer._make_labels(feature_matrix)

    assert labels.name == "distress"
    assert set(labels.unique()).issubset({0, 1})


def test_train_returns_metrics(feature_matrix: pd.DataFrame) -> None:
    scorer = SupplierRiskScorer()
    metrics = scorer.train(feature_matrix)

    assert set(metrics.keys()) == {"accuracy", "roc_auc", "f1"}
    for value in metrics.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_predict_scores_columns(feature_matrix: pd.DataFrame) -> None:
    scorer = SupplierRiskScorer()
    scorer.train(feature_matrix)
    out = scorer.predict_scores(feature_matrix)

    assert list(out.columns) == ["supplier_name", "credit_score", "distress_prob", "risk_label"]
    assert len(out) == len(feature_matrix)


def test_credit_score_range(feature_matrix: pd.DataFrame) -> None:
    scorer = SupplierRiskScorer()
    scorer.train(feature_matrix)
    out = scorer.predict_scores(feature_matrix)

    assert ((out["credit_score"] >= 0.0) & (out["credit_score"] <= 100.0)).all()


def test_risk_label_values(feature_matrix: pd.DataFrame) -> None:
    scorer = SupplierRiskScorer()
    scorer.train(feature_matrix)
    out = scorer.predict_scores(feature_matrix)

    assert set(out["risk_label"].unique()).issubset({"low_risk", "watch", "high_risk"})


def test_guard_trained_raises(feature_matrix: pd.DataFrame) -> None:
    scorer = SupplierRiskScorer()
    with pytest.raises(RuntimeError, match="Model not trained"):
        scorer.predict_scores(feature_matrix)


def test_save_and_load(tmp_path, feature_matrix: pd.DataFrame) -> None:
    scorer = SupplierRiskScorer()
    scorer.train(feature_matrix)

    target = tmp_path / "scorer.joblib"
    saved = scorer.save(target)
    assert saved == target
    assert target.exists()

    loaded = SupplierRiskScorer.load(target)
    assert loaded._is_trained is True

    original_scores = scorer.predict_scores(feature_matrix)["credit_score"].to_numpy()
    loaded_scores = loaded.predict_scores(feature_matrix)["credit_score"].to_numpy()
    assert np.allclose(original_scores, loaded_scores)
