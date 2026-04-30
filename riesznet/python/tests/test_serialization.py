"""save/load round-trip.

Default-MLP path round-trips automatically (functools.partial over a top-level
function pickles by qualname). Custom-factory path requires a top-level function
in the user's own module.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch.nn as nn

from rieszreg import ATE, RieszEstimator, TSM

from riesznet import RieszNet, TorchBackend
from riesznet.modules import build_adam, build_mlp


# ---- Top-level factory used by the custom-architecture round-trip test. ----
# Defining it at module top level so its qualname round-trips.

def _top_level_factory(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 12),
        nn.ReLU(),
        nn.Linear(12, 1),
    )


def test_default_mlp_round_trip(tmp_path, logistic_tsm_df):
    est = RieszNet(
        estimand=TSM(level=1),
        hidden_sizes=(8, 8),
        epochs=15,
        random_state=0,
    )
    est.fit(logistic_tsm_df)
    pred_before = est.predict(logistic_tsm_df)

    save_dir = tmp_path / "fitted"
    est.save(save_dir)

    loaded = RieszNet.load(save_dir)
    pred_after = loaded.predict(logistic_tsm_df)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)


def test_custom_factory_round_trip(tmp_path, logistic_tsm_df):
    backend = TorchBackend(
        module_factory=_top_level_factory,
        optimizer_factory=functools.partial(build_adam, lr=2e-3),
        epochs=15,
    )
    est = RieszEstimator(estimand=TSM(level=1), backend=backend, random_state=0)
    est.fit(logistic_tsm_df)
    pred_before = est.predict(logistic_tsm_df)

    save_dir = tmp_path / "fitted"
    est.save(save_dir)

    loaded = RieszEstimator.load(save_dir)
    pred_after = loaded.predict(logistic_tsm_df)
    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)


def test_metadata_round_trips(tmp_path, logistic_tsm_df):
    import json

    est = RieszNet(
        estimand=TSM(level=1),
        hidden_sizes=(8,),
        epochs=10,
        learning_rate=2e-3,
        random_state=0,
    )
    est.fit(logistic_tsm_df)
    save_dir = tmp_path / "fitted"
    est.save(save_dir)

    with open(save_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["predictor_kind"] == "riesznet"
    assert meta["estimator_class"] == "RieszNet"
    hp = meta["hyperparameters"]
    assert hp["hidden_sizes"] == [8]
    assert hp["learning_rate"] == 2e-3
    assert hp["epochs"] == 10


def test_closure_factory_save_raises(tmp_path, logistic_tsm_df):
    """Defining a factory inside a function should raise on save (qualname carries '<locals>')."""
    def local_factory(input_dim):  # qualname includes "<locals>"
        return nn.Linear(input_dim, 1)

    backend = TorchBackend(
        module_factory=local_factory,
        optimizer_factory=functools.partial(build_adam),
        epochs=2,
    )
    est = RieszEstimator(estimand=TSM(level=1), backend=backend, random_state=0)
    with pytest.raises(ValueError, match="top-level"):
        est.fit(logistic_tsm_df)
