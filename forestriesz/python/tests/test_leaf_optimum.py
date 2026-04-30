"""Single-leaf check: when min_samples_leaf is large enough that no splits
occur, the predicted α equals the closed-form per-leaf optimum:

    locally constant: α* = (Σ m(W_i; 1)) / n
    locally linear:   θ* = (Σ φφ')^{-1} (Σ m(W_i; φ))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rieszreg import trace
from forestriesz import (
    ForestRieszRegressor,
    TSM,
    ATE,
    default_riesz_features,
)


def _moments(rows, estimand, phi_fns):
    """Per-row moments m(W; phi_j), shape (n, p)."""
    feature_keys = estimand.feature_keys
    n = len(rows)
    p = len(phi_fns)
    A = np.zeros((n, p))
    for i, row in enumerate(rows):
        for coef, point in trace(estimand, row):
            point_arr = np.array([[point[k] for k in feature_keys]])
            for j, fn in enumerate(phi_fns):
                A[i, j] += coef * float(fn(point_arr)[0])
    return A


def test_single_basis_leaf_matches_closed_form_tsm():
    """Single-basis sieve [1{T=1}] in one leaf gives θ = A_sum / J_sum."""
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(size=n)
    a = (rng.uniform(size=n) > 0.4).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    estimand = TSM(level=1)
    phi_fns = default_riesz_features(estimand)   # [1{T=1}]
    rows = df.to_dict("records")
    feature_keys = estimand.feature_keys
    features = np.array([[r[k] for k in feature_keys] for r in rows], float)
    phi = np.column_stack([fn(features) for fn in phi_fns])    # (n, 1)
    A = _moments(rows, estimand, phi_fns)
    J = float((phi * phi).sum())
    A_sum = float(A.sum())
    theta_expected = A_sum / J     # closed-form leaf optimum

    est = ForestRieszRegressor(
        estimand=estimand,
        riesz_feature_fns=phi_fns,
        n_estimators=1,
        min_samples_split=10**6,
        min_samples_leaf=10**6,
        max_samples=0.999,
        max_features=None,
        l2=0.0,
        init=0.0,    # closed-form θ = A_sum/J_sum assumes zero base_score
        random_state=0,
    )
    est.fit(df)
    pred = est.predict(df)
    # alpha(z) = θ * 1{T=1}; treated rows predict θ, control rows predict 0.
    treated = df["a"].values == 1
    np.testing.assert_allclose(
        pred[treated], theta_expected, rtol=5e-2, atol=5e-2
    )
    np.testing.assert_allclose(pred[~treated], 0.0, atol=1e-9)


def test_sieve_leaf_matches_closed_form_ate():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-0.5 * x))
    a = (rng.uniform(size=n) < pi).astype(float)
    df = pd.DataFrame({"a": a, "x": x})

    estimand = ATE()
    phi_fns = default_riesz_features(estimand)
    rows = df.to_dict("records")
    feature_keys = estimand.feature_keys
    features = np.array([[r[k] for k in feature_keys] for r in rows], float)
    phi = np.column_stack([fn(features) for fn in phi_fns])    # (n, 2)
    A = _moments(rows, estimand, phi_fns)                       # (n, 2)
    J = phi.T @ phi                                             # (2, 2) summed
    A_sum = A.sum(axis=0)                                       # (2,)
    theta_expected = np.linalg.solve(J, A_sum)                  # (2,)

    est = ForestRieszRegressor(
        estimand=estimand,
        riesz_feature_fns=phi_fns,
        n_estimators=1,
        min_samples_split=10**6,
        min_samples_leaf=10**6,
        max_samples=0.999,
        max_features=None,
        l2=0.0,
        init=0.0,    # ATE has m̄=0 so default is also 0; keep explicit for clarity
        random_state=0,
    )
    est.fit(df)
    # alpha(z) = θ · φ(z) — split features for ATE drop the treatment column,
    # so all rows in the (single, no-split) leaf get the same θ. The
    # prediction depends only on the row's φ.
    pred = est.predict(df)
    expected_pred = (phi * theta_expected[None, :]).sum(axis=1)
    np.testing.assert_allclose(pred, expected_pred, rtol=5e-2, atol=5e-2)
