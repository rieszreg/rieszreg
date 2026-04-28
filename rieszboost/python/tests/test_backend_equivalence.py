"""Backend equivalence: XGBoostBackend and SklearnBackend should produce
statistically equivalent α̂ on the same data with matched hyperparameters.

If the two backends drift apart (beyond what's explainable by sklearn's
exhaustive split-finding vs xgboost's histogram-based finding), one of
them has a bug. This test guards that boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszboost import (
    ATE,
    AdditiveShift,
    ATT,
    RieszBooster,
    SklearnBackend,
    XGBoostBackend,
)


def _logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _binary_df(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = _logit(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x}), pi


def _continuous_df(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2, n)
    a = rng.normal(x**2 - 1.0, np.sqrt(2.0))
    return pd.DataFrame({"a": a, "x": x})


def _truth_ate(df, pi):
    a = df["a"].values
    return a / pi - (1 - a) / (1 - pi)


def _truth_att(df, pi):
    a = df["a"].values
    return a - (1 - a) * pi / (1 - pi)


@pytest.mark.parametrize(
    "name, df_factory, truth_factory, estimand",
    [
        ("ATE", _binary_df, _truth_ate, ATE()),
        ("ATT", _binary_df, _truth_att, ATT()),
    ],
)
def test_xgb_vs_sklearn_backend_predictions_track(
    name, df_factory, truth_factory, estimand,
):
    """Pearson(xgb_alpha, sklearn_alpha) on the same training data should be
    high. Disagreement RMSE should be small relative to truth-RMSE."""
    from sklearn.tree import DecisionTreeRegressor

    df, pi = df_factory(2000, seed=0)
    truth = truth_factory(df, pi)

    # Use gradient_only=True on the xgboost side to match the first-order
    # step that the SklearnBackend uses by construction.
    common = dict(
        n_estimators=200,
        learning_rate=0.05,
        random_state=0,
    )
    xgb_b = RieszBooster(
        estimand=estimand,
        backend=XGBoostBackend(gradient_only=True),
        max_depth=3, reg_lambda=0.0,
        **common,
    ).fit(df)
    skl_b = RieszBooster(
        estimand=estimand,
        backend=SklearnBackend(lambda: DecisionTreeRegressor(max_depth=3, random_state=0)),
        **common,
    ).fit(df)

    xgb_pred = xgb_b.predict(df)
    skl_pred = skl_b.predict(df)

    rmse_xgb = float(np.sqrt(np.mean((xgb_pred - truth) ** 2)))
    rmse_skl = float(np.sqrt(np.mean((skl_pred - truth) ** 2)))
    rmse_diff = float(np.sqrt(np.mean((xgb_pred - skl_pred) ** 2)))
    corr = float(np.corrcoef(xgb_pred, skl_pred)[0, 1])

    msg = (
        f"\n{name}:  RMSE(xgb,truth)={rmse_xgb:.3f}  RMSE(skl,truth)={rmse_skl:.3f}"
        f"  RMSE(xgb,skl)={rmse_diff:.3f}  Pearson={corr:.3f}"
    )
    # Expect strong correlation between backends and disagreement smaller
    # than the typical truth-error (otherwise backends are diverging).
    assert corr > 0.85, msg
    assert rmse_diff < 1.5 * max(rmse_xgb, rmse_skl), msg


def test_xgb_vs_sklearn_additive_shift_continuous():
    """Same equivalence check on a continuous-treatment estimand."""
    from sklearn.tree import DecisionTreeRegressor

    df = _continuous_df(2000, seed=1)
    common = dict(
        estimand=AdditiveShift(delta=1.0),
        n_estimators=200,
        learning_rate=0.05,
        random_state=0,
    )
    xgb_b = RieszBooster(
        backend=XGBoostBackend(gradient_only=True),
        max_depth=3, reg_lambda=0.0, **common,
    ).fit(df)
    skl_b = RieszBooster(
        backend=SklearnBackend(lambda: DecisionTreeRegressor(max_depth=3, random_state=0)),
        **common,
    ).fit(df)
    corr = float(np.corrcoef(xgb_b.predict(df), skl_b.predict(df))[0, 1])
    assert corr > 0.85, f"AdditiveShift backends drifted: Pearson={corr:.3f}"
