"""End-to-end ATE recovery on the linear-Gaussian DGP."""

from __future__ import annotations

import numpy as np
import pytest

from rieszreg.testing import dgps

from riesznet import ATE, RieszNet


def _fit_predict(train, test):
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(32, 32),
        epochs=200,
        learning_rate=5e-3,
        validation_fraction=0.2,
        early_stopping_rounds=30,
        random_state=0,
    )
    est.fit(train)
    return est.predict(test)


def test_ate_consistency_grid():
    rmses = dgps.assert_consistency(
        _fit_predict,
        dgp=dgps.linear_gaussian_ate(),
        n_grid=(400, 1500),
        rng_seed=0,
        tol_at_max_n=1.0,
        monotonicity_slack=0.5,
    )
    # RMSE should drop with sample size (lax check; single-seed noise is real).
    assert rmses[-1] < rmses[0] * 1.5


def test_ate_predict_shape_and_finite(linear_gaussian_ate_df):
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(16, 16),
        epochs=30,
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    pred = est.predict(linear_gaussian_ate_df)
    assert pred.shape == (len(linear_gaussian_ate_df),)
    assert np.all(np.isfinite(pred))


def test_ate_score_is_negative_riesz_loss(linear_gaussian_ate_df):
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(16,),
        epochs=20,
        random_state=0,
    )
    est.fit(linear_gaussian_ate_df)
    score = est.score(linear_gaussian_ate_df)
    loss = est.riesz_loss(linear_gaussian_ate_df)
    assert score == pytest.approx(-loss)


def test_ate_correlation_with_true_alpha():
    """Pearson correlation between α̂ and true α₀ on a single n=1000 fit."""
    dgp = dgps.linear_gaussian_ate()
    rng = np.random.default_rng(0)
    df = dgp.sample(1000, rng)
    est = RieszNet(
        estimand=ATE(),
        hidden_sizes=(32, 32),
        epochs=300,
        learning_rate=5e-3,
        validation_fraction=0.2,
        early_stopping_rounds=30,
        random_state=0,
    )
    est.fit(df)
    alpha_hat = est.predict(df)
    alpha_true = dgp.true_alpha(df)
    # Lax: just confirm we're learning *something* positive.
    corr = np.corrcoef(alpha_hat, alpha_true)[0, 1]
    assert corr > 0.5, f"Pearson corr too low: {corr:.3f}"
