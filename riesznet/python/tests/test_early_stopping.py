"""Early stopping by validation Riesz loss."""

from __future__ import annotations

import pytest

from riesznet import ATE, RieszNet


def test_early_stopping_terminates_early(linear_gaussian_ate_df):
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(8,),
        epochs=200,
        learning_rate=5e-3,
        validation_fraction=0.2,
        early_stopping_rounds=3,
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    # With patience=3 on a tiny model, fitting should stop well before 200 epochs.
    assert est.best_iteration_ is not None
    assert est.best_iteration_ < 200


def test_early_stopping_without_validation_fraction_raises(linear_gaussian_ate_df):
    """early_stopping_rounds without validation_fraction>0 raises a clear
    error: the orchestrator no longer auto-splits."""
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(8,),
        epochs=10,
        early_stopping_rounds=3,
        validation_fraction=0.0,
        random_state=0,
    )
    with pytest.raises(ValueError, match="validation"):
        est.fit(linear_gaussian_ate_df)
