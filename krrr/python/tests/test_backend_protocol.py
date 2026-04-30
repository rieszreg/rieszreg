"""KernelRidgeBackend satisfies rieszreg's Backend / Predictor protocols."""

from __future__ import annotations

import numpy as np

from rieszreg.backends.base import Backend, FitResult, Predictor

from krrr import KernelRidgeBackend, KernelPredictor


def test_backend_has_fit_augmented():
    """Structural Protocol check (rieszreg.Backend isn't @runtime_checkable)."""
    backend = KernelRidgeBackend()
    assert hasattr(backend, "fit_augmented")
    assert callable(backend.fit_augmented)


def test_predictor_is_a_predictor():
    # Sufficient to satisfy the Protocol — `predict_eta` and `predict_alpha`
    # have the right signatures. We can't instantiate without a SolveResult,
    # but Protocol membership is structural; we check the methods exist.
    assert hasattr(KernelPredictor, "predict_eta")
    assert hasattr(KernelPredictor, "predict_alpha")


def test_backend_via_riesz_estimator(binary_ate_data):
    """RieszEstimator(backend=KernelRidgeBackend(...)) fits and predicts."""
    from rieszreg import ATE, RieszEstimator

    df, _, _ = binary_ate_data
    est = RieszEstimator(
        estimand=ATE("a", ("x",)),
        backend=KernelRidgeBackend(
            lambda_grid=np.logspace(-3, 0, 6),
            validation_fraction=0.25,
        ),
    )
    est.fit(df)
    alpha_hat = est.predict(df)
    assert alpha_hat.shape == (len(df),)
    assert np.all(np.isfinite(alpha_hat))
    # Predictor type
    assert est.predictor_.kind == "krrr"
    # Riesz loss is a finite number
    assert np.isfinite(est.riesz_loss(df))
