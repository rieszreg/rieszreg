"""An array eval_set after a DataFrame Z is read in feature_keys order, which
need not be Z's column order, so fit warns."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from rieszreg import ATE, FitResult, RieszEstimator


@dataclass
class _Const:
    def predict_eta(self, X):
        return np.zeros(len(X))

    def predict_alpha(self, X):
        return np.zeros(len(X))


class _AugBackend:
    def fit_augmented(self, aug_train, aug_valid, loss, **kwargs):
        return FitResult(predictor=_Const())


def _df(n=60):
    rng = np.random.default_rng(0)
    # Columns out of feature_keys order: feature_keys is ("a", "w", "x").
    return pd.DataFrame({
        "w": rng.binomial(1, 0.5, n).astype(float),
        "a": rng.binomial(1, 0.5, n).astype(float),
        "x": rng.normal(size=n),
    })


def test_array_eval_set_after_dataframe_warns():
    df = _df()
    est = RieszEstimator(estimand=ATE(treatment="a", covariates=["w", "x"]), backend=_AugBackend())
    with pytest.warns(UserWarning, match="eval_set is an array"):
        est.fit(df, eval_set=df.to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est.fit(df, eval_set=df)
