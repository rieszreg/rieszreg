"""Parity vs Lee & Schuler's reference implementation (kaitlynjlee/boosting_for_rr).

Inlines Lee's `fit_internal` for the ATE and ATT estimators verbatim, so this
test stays self-contained — no external repo to clone in CI. Verifies that
rieszboost with `gradient_only=True, learning_rate=lr_ref/2, reg_lambda=0`
reproduces the reference at the Pearson correlation documented in
`examples/lee_schuler/COMPARISON.md` (ATE: 0.998, ATT: 0.986).

Both ATE_ES_stochastic.fit_internal and ATT_ES_stochastic.fit_internal in the
reference use the same gradient-boosting loop:
    residuals = D * (f - linear_orig(a_aug)) + (1 - D) * linear_cf
    f -= learning_rate * tree.predict(data)
The "(1 - 2 * data[:, 0])" expression at counterfactual rows reads the
*augmented* treatment column (which is 1 - a_i at counterfactual rows of the
ATE augmentation). The data layouts differ between ATE and ATT — see below.

Reference: github.com/kaitlynjlee/boosting_for_rr (rrboost/boosters.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone as sklearn_clone
from sklearn.tree import DecisionTreeRegressor

from rieszboost import RieszBooster, XGBoostBackend
from rieszreg import ATE, ATT


# ---- DGP (same as examples/lee_schuler/binary_dgp.py) ----

def _binary_dgp(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = 1.0 / (1.0 + np.exp(-(-0.02 * x - x ** 2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi).astype(float)
    return a, x


# ---- Lee-Schuler reference (literal copy of fit_internal / predict_internal) ----

def _ate_lee_reference(
    a: np.ndarray, x: np.ndarray, *,
    learning_rate: float, n_estimators: int, max_depth: int, random_state: int,
) -> np.ndarray:
    """Verbatim reproduction of `ATE_ES_stochastic.fit_internal` + `predict`.

    Augmentation: each subject contributes one original row and one
    counterfactual at the flipped treatment. Reference layout:
        data = [(A_i, X_i)]_i    +
               [(0, X_i) for i with A_i = 1]   +
               [(1, X_i) for i with A_i = 0]
    """
    A = a.reshape(-1, 1)
    X = x.reshape(-1, 1)
    n = len(A)

    data = np.row_stack((
        np.column_stack((A, X)),
        np.column_stack((np.zeros(n)[A[:, 0] == 1], X[A[:, 0] == 1, :])),
        np.column_stack((np.ones(n)[A[:, 0] == 0], X[A[:, 0] == 0, :])),
    ))
    D = np.concatenate((np.ones(n), np.zeros(n)))

    f = np.zeros(data.shape[0])
    learners = []
    base = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    for _ in range(n_estimators):
        residuals = D * (f - (2 * data[:, 0] - 1)) + (1 - D) * (1 - 2 * data[:, 0])
        learner = sklearn_clone(base)
        learner.fit(data, residuals)
        f -= learning_rate * learner.predict(data)
        learners.append(learner)

    # Predict α̂ at the original (a, x) rows: sum of -lr * learner.predict.
    pred_data = np.column_stack((a, x))
    alpha = sum(-learning_rate * lrn.predict(pred_data) for lrn in learners)
    return np.asarray(alpha)


def _att_lee_reference(
    a: np.ndarray, x: np.ndarray, *,
    learning_rate: float, n_estimators: int, max_depth: int, random_state: int,
) -> np.ndarray:
    """Verbatim reproduction of `ATT_ES_stochastic.fit_internal` + `predict`.

    Reference layout: counterfactuals only for treated subjects.
        data = [(A_i, X_i)]_i    +
               [(0, X_i) for i with A_i = 1]
        D    = ones(n) ++ zeros(#treated)
    """
    A = a
    X = x.reshape(-1, 1)
    n = len(A)

    data = np.row_stack((
        np.column_stack((A.reshape(-1, 1), X)),
        np.column_stack((np.zeros(n)[A == 1].reshape(-1, 1), X[A == 1, :])),
    ))
    D = np.concatenate((np.ones(n), np.zeros(int((A == 1).sum()))))

    f = np.zeros(data.shape[0])
    learners = []
    base = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    for _ in range(n_estimators):
        residuals = D * (f - data[:, 0]) + (1 - D) * 1.0
        learner = sklearn_clone(base)
        learner.fit(data, residuals)
        f -= learning_rate * learner.predict(data)
        learners.append(learner)

    pred_data = np.column_stack((a, x))
    alpha = sum(-learning_rate * lrn.predict(pred_data) for lrn in learners)
    return np.asarray(alpha)


# ---- Tests ----

@pytest.mark.parametrize(
    "estimand_name,factory,reference_fn,pearson_floor",
    [
        ("ATE", ATE, _ate_lee_reference, 0.95),
        ("ATT", ATT, _att_lee_reference, 0.85),
    ],
)
def test_rieszboost_matches_lee_schuler(
    estimand_name, factory, reference_fn, pearson_floor
):
    """rieszboost(gradient_only=True, lr_ref/2, reg_lambda=0) ≈ Lee's reference.

    COMPARISON.md reports Pearson 0.998 (ATE) and 0.986 (ATT) on a 10-seed
    average. We assert > 0.95 / 0.85 on a single seed to leave headroom for
    seed sensitivity and tree-backend differences (xgboost histogram splits
    vs sklearn exact splits) without becoming a flaky test.
    """
    n = 500
    seed = 0
    lr_ref = 0.1
    n_estimators = 100
    max_depth = 3

    a, x = _binary_dgp(n, seed)
    df = pd.DataFrame({"a": a, "x": x})

    alpha_ref = reference_fn(
        a, x,
        learning_rate=lr_ref,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=0,
    )

    booster = RieszBooster(
        estimand=factory(treatment="a", covariates=("x",)),
        backend=XGBoostBackend(gradient_only=True),
        learning_rate=lr_ref / 2.0,
        n_estimators=n_estimators,
        max_depth=max_depth,
        reg_lambda=0.0,
        random_state=0,
        init=0.0,
    ).fit(df)
    alpha_ours = booster.predict(df)

    pearson = float(np.corrcoef(alpha_ref, alpha_ours)[0, 1])
    rmse_diff = float(np.sqrt(np.mean((alpha_ref - alpha_ours) ** 2)))
    rmse_ref_scale = float(np.sqrt(np.mean(alpha_ref ** 2)))

    assert pearson >= pearson_floor, (
        f"{estimand_name}: Pearson(ours, Lee_ref) = {pearson:.4f} < {pearson_floor}; "
        "rieszboost may have drifted from the Lee-Schuler algorithm."
    )
    assert rmse_diff < rmse_ref_scale, (
        f"{estimand_name}: ref-vs-ours RMSE ({rmse_diff:.3f}) >= ref scale "
        f"({rmse_ref_scale:.3f}) — predictions are no longer comparable."
    )
