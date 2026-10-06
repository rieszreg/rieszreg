"""The orchestrator holds out the fraction of rows the backend asks for.

A backend with `holdout_fraction()` decides for each fit; one without it gets
no holdout, even with a `validation_fraction` attribute. An `eval_set` passed to
`fit` replaces the split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg import ATE, FitResult, RieszEstimator
from rieszreg.backends import holdout_fraction


class _ConstPredictor:
    kind = "holdout-test"

    def predict_eta(self, features):
        return np.zeros(len(features))

    def predict_alpha(self, features):
        return self.predict_eta(features)

    def save(self, dir_path):  # pragma: no cover - not used
        pass


class _Recorder:
    """Records the original-row counts of the train and validation data."""

    n_train = n_valid = None

    def fit_augmented(self, aug_train, aug_valid, loss, **kwargs):
        self.n_train = aug_train.n_rows
        self.n_valid = None if aug_valid is None else aug_valid.n_rows
        return FitResult(predictor=_ConstPredictor())


class _MethodBackend(_Recorder):
    def __init__(self, fraction, validation_fraction=0.5):
        self.fraction = fraction
        self.validation_fraction = validation_fraction  # ignored: the method wins

    def holdout_fraction(self):
        return self.fraction


class _AttributeBackend(_Recorder):
    def __init__(self, validation_fraction):
        self.validation_fraction = validation_fraction


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.binomial(1, 0.5, n).astype(float), "x": rng.uniform(size=n)})


def _fit(backend, df, **kwargs):
    RieszEstimator(estimand=ATE(), backend=backend).fit(df, **kwargs)
    return backend


def test_method_sets_the_holdout():
    b = _fit(_MethodBackend(0.25), _df())
    assert (b.n_train, b.n_valid) == (150, 50)


def test_method_returning_zero_holds_nothing_out():
    """The method wins over a validation_fraction attribute."""
    b = _fit(_MethodBackend(0.0, validation_fraction=0.5), _df())
    assert (b.n_train, b.n_valid) == (200, None)


def test_validation_fraction_attribute_alone_holds_nothing_out():
    b = _fit(_AttributeBackend(0.2), _df())
    assert (b.n_train, b.n_valid) == (200, None)


def test_no_holdout_without_method_or_attribute():
    b = _fit(_Recorder(), _df())
    assert (b.n_train, b.n_valid) == (200, None)
    assert holdout_fraction(_Recorder()) == 0.0


def test_eval_set_replaces_the_split():
    b = _fit(_MethodBackend(0.25), _df(), eval_set=_df(n=30, seed=1))
    assert (b.n_train, b.n_valid) == (200, 30)


@pytest.mark.parametrize("backend, expected", [
    (_MethodBackend(0.3), 0.3),
    (_AttributeBackend(0.2), 0.0),
])
def test_holdout_fraction_helper(backend, expected):
    assert holdout_fraction(backend) == expected
