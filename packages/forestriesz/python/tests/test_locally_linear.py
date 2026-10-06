"""Locally linear sieve fit recovers IPW representer for ATE.

The forest splits on covariates; the [1{T=0}, 1{T=1}] sieve resolves
treatment within each leaf. With enough trees the per-leaf solution
converges to the inverse propensity weights.
"""

from __future__ import annotations

import numpy as np

from forestriesz import ATE, ForestRieszRegressor
from forestriesz.feature_fns import default_riesz_features


def test_sieve_recovers_ipw_signs(linear_gaussian_ate_df):
    df = linear_gaussian_ate_df
    est = ForestRieszRegressor(
        estimand=ATE(),
        riesz_feature_fns=default_riesz_features(ATE()),
        n_estimators=200,
        min_samples_leaf=20,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    a = df["a"].values
    # IPW representer is +1/p(1|x) for treated, -1/p(0|x) for control —
    # so sign(α̂) should align with (2T − 1) for the vast majority of rows.
    expected_sign = 2 * a - 1
    matches = np.sign(pred) == expected_sign
    assert matches.mean() > 0.85, (
        f"Sign agreement {matches.mean():.3f} too low; "
        "sieve fit isn't separating treatment arms."
    )


def test_sieve_works_for_tsm(logistic_tsm_df):
    """The auto sieve [1{T=level}] is what makes TSM learnable."""
    from forestriesz import TSM

    df = logistic_tsm_df
    est = ForestRieszRegressor(
        estimand=TSM(level=1.0),
        n_estimators=100,
        min_samples_leaf=15,
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    a = df["a"].values
    # α₀ = 1{T=1} / π(x); for T=0 rows truth is exactly 0, for T=1 rows it's positive.
    assert np.all(pred[a == 0] >= -1e-9)
    assert pred[a == 1].mean() > 0.5


def test_split_features_default_drops_treatment_for_ate():
    from forestriesz.feature_fns import default_split_feature_indices

    idx = default_split_feature_indices(ATE(covariates=["x"]), default_riesz_features(ATE()))
    # ATE feature_keys = ('a', 'x'); with treatment-indexed sieve, splitter sees only 'x'.
    assert idx == (1,)


def test_split_features_default_uses_all_when_no_sieve():
    from forestriesz.feature_fns import default_split_feature_indices

    idx = default_split_feature_indices(ATE(covariates=["x"]), None)
    assert idx == (0, 1)
