"""Backend Protocol satisfaction and orchestrator dispatch."""
from __future__ import annotations

import numpy as np

from rieszreg import ATE, FitResult, RieszEstimator
from riesztree import RieszTreeBackend, RieszTreePredictor, RieszTreeRegressor


def test_backend_returns_fitresult(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(500, seed=0)
    estimand = ATE(treatment="a", covariates=covariate_keys)
    feats = df[["a", *covariate_keys]].to_numpy(dtype=float)
    aug = estimand.augment(feats)
    from rieszreg import SquaredLoss
    out = RieszTreeBackend(max_depth=3).fit_augmented(
        aug, None, SquaredLoss(),
        base_score=0.0, random_state=0, hyperparams={},
    )
    assert isinstance(out, FitResult)
    assert isinstance(out.predictor, RieszTreePredictor)
    pred = out.predictor.predict_alpha(aug.features)
    assert pred.shape == (aug.features.shape[0],)


def test_estimator_dispatches_to_augmented(linear_gaussian_ate, covariate_keys):
    """RieszEstimator should recognise RieszTreeBackend as augmentation-style
    and call ``fit_augmented`` (not ``fit_rows``)."""
    make, truth = linear_gaussian_ate
    df = make(500, seed=0)
    estimand = ATE(treatment="a", covariates=covariate_keys)
    est = RieszEstimator(estimand=estimand, backend=RieszTreeBackend(max_depth=4))
    est.fit(df)
    a_hat = est.predict(df)
    assert a_hat.shape == (500,)
    # Sanity: pearson with truth > 0.5 even at small n.
    assert np.corrcoef(a_hat, truth(df))[0, 1] > 0.5


def test_convenience_class_works(linear_gaussian_ate, covariate_keys):
    make, truth = linear_gaussian_ate
    df = make(800, seed=1)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=4,
    )
    est.fit(df)
    a_hat = est.predict(df)
    assert np.corrcoef(a_hat, truth(df))[0, 1] > 0.85
