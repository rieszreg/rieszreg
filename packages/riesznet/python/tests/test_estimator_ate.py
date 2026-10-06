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


def test_standardize_makes_fit_invariant_to_covariate_scale():
    """With standardize=True, an affine change of units in a covariate leaves
    the fitted α̂ unchanged, because the counterfactual rows are standardized
    with the observed rows' mean and scale."""
    df = dgps.linear_gaussian_ate().sample(300, np.random.default_rng(0))
    rescaled = df.assign(x=1000.0 * df["x"] + 50.0)
    kw = dict(estimand=ATE(), hidden_sizes=(16,), epochs=30, dtype="float64", random_state=0)
    np.testing.assert_allclose(
        RieszNet(**kw).fit(df).predict(df),
        RieszNet(**kw).fit(rescaled).predict(rescaled),
        rtol=1e-6, atol=1e-8,
    )
    # Without standardization the raw scale reaches the network.
    raw = RieszNet(**kw, standardize=False)
    assert not np.allclose(raw.fit(df).predict(df), raw.fit(rescaled).predict(rescaled), rtol=1e-3)


def test_standardize_leaves_a_constant_column_unscaled():
    """A constant 0.1 has a float std of about 1e-16. Dividing by it would
    send any other value of that column to ~1e15 at predict time."""
    df = dgps.linear_gaussian_ate().sample(300, np.random.default_rng(0)).assign(c=0.1)
    est = RieszNet(estimand=ATE(covariates=["x", "c"]), hidden_sizes=(16,), epochs=5,
                   random_state=0).fit(df)
    shifted = est.predict(df.assign(c=0.2))
    assert np.abs(shifted).max() < 100 * np.abs(est.predict(df)).max()
