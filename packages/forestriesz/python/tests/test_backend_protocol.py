"""ForestRieszBackend satisfies MomentBackend; ForestPredictor satisfies Predictor."""

from __future__ import annotations


import forestriesz
from forestriesz import ForestRieszBackend
from forestriesz.predictor import ForestPredictor


def test_backend_exposes_fit_rows_only():
    backend = ForestRieszBackend()
    assert callable(getattr(backend, "fit_rows", None))
    # Moment-style: must NOT advertise fit_augmented (that's how the orchestrator dispatches)
    assert not hasattr(backend, "fit_augmented")


def test_predictor_protocol_surface():
    assert ForestPredictor.kind == "forestriesz"
    for attr in ("predict_eta", "predict_alpha", "save"):
        assert callable(getattr(ForestPredictor, attr)), attr


def test_predictor_loader_registered():
    from rieszreg.backends.base import _PREDICTOR_LOADERS

    assert "forestriesz" in _PREDICTOR_LOADERS, (
        "Importing forestriesz must register the predictor loader. "
        "Check that predictor.py runs `register_predictor_loader('forestriesz', ...)`."
    )


def test_namespace_reexports_shared_user_api():
    from rieszreg import user_api

    assert set(user_api.__all__) <= set(forestriesz.__all__)
    assert {"ForestRieszRegressor", "AugForestRieszRegressor"} <= set(forestriesz.__all__)
