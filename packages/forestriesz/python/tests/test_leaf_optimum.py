"""Single-leaf check: when min_samples_leaf is large enough that no splits
occur, the predicted α equals the closed-form per-leaf optimum:

    locally constant: α* = (Σ m(W_i; 1)) / n
    locally linear:   θ* = (Σ φφ')^{-1} (Σ m(W_i; φ))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rieszreg import trace
from forestriesz import ForestRieszRegressor, TSM, ATE
from forestriesz.feature_fns import default_riesz_features


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

    estimand = TSM(level=1, covariates=["x"])
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

    estimand = ATE(covariates=["x"])
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


def test_vectorized_moments_match_trace():
    """The backend reads m(W_i; φ_j) off ``estimand.augment``; it must equal
    the per-row trace Σ coef · φ_j(point) for every built-in and a custom
    estimand."""
    from rieszreg import (
        ATT, AdditiveShift, FiniteEvalEstimand, LocalShift, OutcomeRegNormSq,
    )
    from forestriesz.backend import _per_row_moments

    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({"a": rng.choice([0.0, 1.0], n), "x": rng.normal(size=n)})
    y = rng.normal(size=n)
    phi_fns = [lambda f: np.ones(len(f)), lambda f: f[:, 0], lambda f: f[:, 0] * f[:, 1] ** 2]
    custom = FiniteEvalEstimand(
        feature_keys=("a", "x"),
        m=lambda alpha: lambda z, y=None: 2.0 * alpha(a=1, x=z["x"]) - alpha(a=z["a"], x=0.5 * z["x"]),
    )
    for estimand, ys in [
        (ATE(covariates=["x"]), None), (ATT(covariates=["x"]), None),
        (TSM(level=0, covariates=["x"]), None), (AdditiveShift(delta=0.7, covariates=["x"]), None),
        (LocalShift(delta=0.3, threshold=0.5, covariates=["x"]), None), (custom, None),
        (OutcomeRegNormSq(covariates=["a", "x"]), y),
    ]:
        rows = df.to_dict("records")
        X = df[list(estimand.feature_keys)].to_numpy(float)
        reference = np.zeros((n, len(phi_fns)))
        for i, row in enumerate(rows):
            for coef, point in trace(estimand, row, None if ys is None else ys[i]):
                pt = np.array([[point[k] for k in estimand.feature_keys]], float)
                reference[i] += coef * np.array([fn(pt)[0] for fn in phi_fns])
        got = _per_row_moments(estimand.augment(X, ys=ys), phi_fns)
        np.testing.assert_allclose(got, reference, atol=1e-12, err_msg=estimand.name)
