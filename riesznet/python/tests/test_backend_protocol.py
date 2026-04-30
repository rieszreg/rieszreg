"""TorchBackend satisfies MomentBackend; TorchPredictor satisfies Predictor."""

from __future__ import annotations

import functools

import numpy as np

import riesznet
from riesznet import RieszNet, TorchBackend, TorchPredictor
from riesznet.modules import build_adam, build_mlp


def test_backend_exposes_fit_rows_only():
    backend = TorchBackend(
        module_factory=functools.partial(build_mlp),
        optimizer_factory=functools.partial(build_adam),
    )
    assert callable(getattr(backend, "fit_rows", None))
    # Moment-style: must NOT advertise fit_augmented (orchestrator dispatches on this).
    assert not hasattr(backend, "fit_augmented")


def test_predictor_protocol_surface():
    assert TorchPredictor.kind == "riesznet"
    for attr in ("predict_eta", "predict_alpha", "save"):
        assert callable(getattr(TorchPredictor, attr)), attr


def test_predictor_loader_registered():
    from rieszreg.backends.base import _PREDICTOR_LOADERS

    assert "riesznet" in _PREDICTOR_LOADERS, (
        "Importing riesznet must register the predictor loader. "
        "Check that backend.py runs `register_predictor_loader('riesznet', ...)`."
    )


def test_namespace_exports_match_design():
    expected = {
        "RieszNet",
        "TorchBackend",
        "TorchPredictor",
        "ATE",
        "ATT",
        "AdditiveShift",
        "BernoulliLoss",
        "BoundedSquaredLoss",
        "Diagnostics",
        "Estimand",
        "KLLoss",
        "LinearForm",
        "LocalShift",
        "LossSpec",
        "SquaredLoss",
        "StochasticIntervention",
        "TSM",
        "Tracer",
        "trace",
    }
    assert expected.issubset(set(riesznet.__all__))


def test_rieszreg_orchestrator_dispatches_to_fit_rows(linear_gaussian_ate_df):
    """End-to-end: composing TorchBackend with rieszreg.RieszEstimator works."""
    from rieszreg import ATE, RieszEstimator

    backend = TorchBackend(
        module_factory=functools.partial(build_mlp, hidden_sizes=(16,)),
        optimizer_factory=functools.partial(build_adam, lr=1e-2),
        epochs=20,
    )
    est = RieszEstimator(estimand=ATE(), backend=backend, random_state=0)
    est.fit(linear_gaussian_ate_df)
    pred = est.predict(linear_gaussian_ate_df)
    assert pred.shape == (len(linear_gaussian_ate_df),)
    assert np.all(np.isfinite(pred))
