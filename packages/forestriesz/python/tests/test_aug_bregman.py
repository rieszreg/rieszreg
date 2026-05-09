"""AugForestRieszRegressor with non-quadratic Bregman losses.

The augmentation-style backend handles all four built-in losses via the
riesztree-backed loss-aware splitter — no post-hoc Newton, no per-loss
configuration on the forest side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
)
from rieszreg.testing import dgps

from forestriesz import (
    AugForestRieszRegressor,
    ATE,
    TSM,
)


# ---- end-to-end: predictions stay in the loss's natural domain ------------


@pytest.fixture
def df_binary():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-0.5 * x))
    a = (rng.uniform(size=n) < pi).astype(float)
    return pd.DataFrame({"a": a, "x": x}), x, pi, a


def test_kl_loss_predictions_in_natural_domain(df_binary):
    """KL's natural α-domain is the non-negative reals. TSM's truth attains
    α₀ = 0 at A=0 inputs, so a correctly fit forest should return values in
    the closure of the natural domain (≥ 0)."""
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=TSM(level=1),
        loss=KLLoss(),
        n_estimators=50,
        min_samples_leaf=10,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    assert pred.shape == (len(df),)
    assert np.all(pred >= 0)
    assert np.all(np.isfinite(pred))
    assert np.any(pred > 0)


def test_bernoulli_loss_predictions_in_natural_domain(df_binary):
    """Bernoulli's natural α-domain is [0, 1]. TSM's truth attains α₀ = 0 at
    A=0 inputs, so values at the lower boundary are expected."""
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=TSM(level=1),
        loss=BernoulliLoss(),
        n_estimators=50,
        min_samples_leaf=10,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    assert np.all((pred >= 0) & (pred <= 1))
    assert np.any((pred > 0) & (pred < 1))


def test_bounded_squared_predictions_in_bounds(df_binary):
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=ATE(),
        loss=BoundedSquaredLoss(lo=-15.0, hi=15.0),
        n_estimators=50,
        min_samples_leaf=10,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    assert np.all((pred > -15.0) & (pred < 15.0)), "BoundedSquaredLoss must clip to its bounds"


# ---- consistency: KL converges to the true IPW representer ----------------


def test_kl_converges_to_truth_on_tsm():
    """KLLoss with TSM should recover α₀ = 1{T=1}/π(X) on the logistic_tsm DGP."""
    def fit_predict(train, test):
        est = AugForestRieszRegressor(
            estimand=TSM(level=1.0),
            loss=KLLoss(),
            n_estimators=200,
            min_samples_leaf=10,
            random_state=0,
        )
        est.fit(train)
        return est.predict(test)

    rmses = dgps.assert_consistency(
        fit_predict,
        dgp=dgps.logistic_tsm(level=1.0),
        n_grid=(500, 2000),
        rng_seed=0,
        tol_at_max_n=1.0,
        monotonicity_slack=0.5,
    )
    assert rmses[-1] < rmses[0] * 1.5


# ---- save/load round-trips through the per-tree predictor.json files ------


def test_save_load_round_trip_preserves_kl_predictions(tmp_path, df_binary):
    df, _, _, _ = df_binary
    est = AugForestRieszRegressor(
        estimand=TSM(level=1),
        loss=KLLoss(),
        n_estimators=30,
        min_samples_leaf=15,
        random_state=0,
    )
    est.fit(df)
    pred_before = est.predict(df)

    save_dir = tmp_path / "kl_fit"
    est.save(save_dir)

    loaded = AugForestRieszRegressor.load(save_dir)
    pred_after = loaded.predict(df)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-12)
