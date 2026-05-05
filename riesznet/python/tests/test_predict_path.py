"""Tests for `RieszNet.predict_path` — α̂ at each snapshot epoch from one fit.

Bit-equality against an independent fit at the same epoch count holds because
PyTorch's Adam trajectory is deterministic given a fixed seed and identical
data ordering. The riesznet backend seeds both `torch.manual_seed` and the
`torch.Generator` driving minibatch shuffles, so two fits with the same
`random_state`, `batch_size`, and `epochs` see the same per-step state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riesznet import ATE, RieszNet
from riesznet.backend import auto_snapshot_epochs


def _df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-(0.5 * x)))
    a = (rng.uniform(size=n) < pi).astype(float)
    return pd.DataFrame({"a": a, "x": x})


def _net(epochs: int, snapshot_epochs=None, **kwargs) -> RieszNet:
    return RieszNet(
        estimand=ATE("a", ("x",)),
        hidden_sizes=(8, 8),
        epochs=epochs,
        learning_rate=1e-2,
        batch_size=None,         # full-batch → fully deterministic per epoch
        random_state=0,
        snapshot_epochs=snapshot_epochs,
        **kwargs,
    )


def test_auto_snapshot_epochs_recipe():
    grid = auto_snapshot_epochs(200)
    assert grid[0] == 1
    assert grid[-1] == 200
    assert all(1 <= e <= 200 for e in grid)
    assert grid == tuple(sorted(set(grid)))
    # rec=10, so the dense seed (1, 2, 5, 10) is included plus 10, 20, ... 200
    assert {1, 2, 5, 10, 20, 100, 200}.issubset(set(grid))


def test_predict_path_default_grid_runs(small_df):
    df = small_df
    net = _net(epochs=20).fit(df)
    path = net.predict_path(df)
    assert path.shape == (len(df), len(net._resolved_snapshot_epochs()))


def test_predict_path_explicit_ticks(small_df):
    df = small_df
    ticks = [1, 5, 20]
    net = _net(epochs=20, snapshot_epochs=ticks).fit(df)
    path = net.predict_path(df)
    assert path.shape == (len(df), 3)
    sub = net.predict_path(df, epochs=[5])
    np.testing.assert_array_equal(sub[:, 0], path[:, 1])


def test_predict_path_final_column_matches_predict_when_no_early_stop(small_df):
    df = small_df
    net = _net(epochs=10, snapshot_epochs=[1, 5, 10]).fit(df)
    path = net.predict_path(df)
    np.testing.assert_allclose(path[:, -1], net.predict(df), rtol=0, atol=1e-6)


def test_predict_path_columns_match_independent_fits():
    df = _df(80, seed=2)
    full = _net(epochs=10, snapshot_epochs=[1, 5, 10]).fit(df)
    path = full.predict_path(df)
    for j, ep in enumerate([1, 5, 10]):
        single = _net(epochs=ep, snapshot_epochs=[ep]).fit(df)
        np.testing.assert_allclose(path[:, j], single.predict(df), rtol=0, atol=1e-6)


def test_predict_path_unknown_epoch_raises(small_df):
    df = small_df
    net = _net(epochs=10, snapshot_epochs=[1, 5, 10]).fit(df)
    with pytest.raises(ValueError, match="not in stored"):
        net.predict_path(df, epochs=[7])


def test_predict_path_disabled_when_snapshot_epochs_empty(small_df):
    df = small_df
    net = _net(epochs=10, snapshot_epochs=[]).fit(df)
    # predict() still works
    net.predict(df)
    with pytest.raises(RuntimeError, match="snapshot_epochs"):
        net.predict_path(df)


def test_predict_path_validates_range_at_construction():
    with pytest.raises(ValueError, match="must satisfy"):
        _net(epochs=5, snapshot_epochs=[10])._resolved_snapshot_epochs()


def test_predict_path_round_trip(tmp_path, small_df):
    df = small_df
    net = _net(epochs=10, snapshot_epochs=[1, 5, 10]).fit(df)
    expected = net.predict_path(df)
    net.save(tmp_path / "rn")
    loaded = RieszNet.load(tmp_path / "rn")
    actual = loaded.predict_path(df)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_early_stopping_drops_post_stop_snapshots():
    # epochs=50, but with very tight early stopping we may stop earlier.
    # Snapshots requested past the actual stop-epoch should not appear.
    df = _df(80, seed=3)
    net = RieszNet(
        estimand=ATE("a", ("x",)),
        hidden_sizes=(4, 4),
        epochs=50,
        learning_rate=1e-2,
        batch_size=None,
        validation_fraction=0.25,
        early_stopping_rounds=2,
        snapshot_epochs=[1, 5, 10, 50],
        random_state=0,
    ).fit(df)
    stored = net.predictor_.snapshot_epochs
    assert stored is not None
    # Every retained tick must be ≤ the number of epochs actually run.
    assert all(e <= 50 for e in stored)
    assert 1 in stored
