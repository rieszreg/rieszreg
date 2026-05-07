"""ForestRieszBackend satisfies MomentBackend; ForestPredictor satisfies Predictor."""

from __future__ import annotations

import numpy as np

import forestriesz
from forestriesz import ForestPredictor, ForestRieszBackend


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


def test_namespace_exports_match_design():
    expected = {
        "ATE",
        "ATT",
        "AdditiveShift",
        "ForestDiagnostics",
        "ForestPredictor",
        "ForestRieszBackend",
        "ForestRieszRegressor",
        "LocalShift",
        "Loss",
        "SquaredLoss",
        "TSM",
        "default_riesz_features",
        "default_split_feature_indices",
        "diagnose_forest",
    }
    assert expected.issubset(set(forestriesz.__all__))
