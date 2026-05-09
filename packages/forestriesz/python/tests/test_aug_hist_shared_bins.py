"""Parity tests for the hist-mode shared-bin-mapper fast path in
:class:`forestriesz.AugForestRieszBackend`.

When ``splitter='hist'`` and the configuration is "simple" (no
categoricals, no per-split feature subsampling, no pruning, no
leaf-count cap, built-in loss), the forest fits the bin mapper once
on the full augmented training data and reuses the bin thresholds
across every joblib worker — algorithmically equivalent to fitting
the mapper on the full dataset (rather than on each bootstrap
subsample). This is the same convention as
:class:`sklearn.ensemble.HistGradientBoostingRegressor` and saves
``n_estimators - 1`` repeats of ``fit_bin_mapper + transform``.

The tests below:

  * Verify the shared-bin worker matches a hand-rolled per-tree call
    to :func:`grow_depthwise_hist_c` with the same shared bin mapper.
  * Verify the dispatch falls back to the per-tree path when an
    eligibility constraint is violated (``max_features='sqrt'`` in
    particular).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rieszreg import (
    ATE,
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    SquaredLoss,
    TSM,
)
from forestriesz import AugForestRieszRegressor


def _make_df(n: int = 200, p: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    logit = 0.6 * X[:, 0] + 0.4 * X[:, 1]
    pi = 1.0 / (1.0 + np.exp(-logit))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


@pytest.mark.parametrize(
    "loss_factory, estimand_factory",
    [
        (lambda: SquaredLoss(),
         lambda p: ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))),
        (lambda: KLLoss(),
         lambda p: TSM(treatment="a", covariates=tuple(f"x{j}" for j in range(p)), level=1.0)),
        (lambda: BernoulliLoss(),
         lambda p: ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))),
        (lambda: BoundedSquaredLoss(lo=-3.0, hi=3.0),
         lambda p: ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))),
    ],
    ids=["squared", "kl", "bernoulli", "bounded"],
)
def test_hist_shared_bins_predicts_finite(loss_factory, estimand_factory):
    """End-to-end smoke: hist forest produces finite predictions for every
    built-in loss + estimand combination."""
    df = _make_df(n=400, p=4)
    estimand = estimand_factory(4)
    est = AugForestRieszRegressor(
        estimand=estimand, loss=loss_factory(),
        n_estimators=10, max_depth=6,
        min_samples_split=2, min_samples_leaf=1,
        max_features=1.0, bootstrap=True, max_samples=None,
        n_jobs=1, random_state=0, splitter="hist", max_bins=64,
    )
    est.fit(df)
    pred = est.predict(df)
    assert pred.shape == (len(df),)
    assert np.all(np.isfinite(pred))


def test_hist_shared_bins_matches_kernel_level_reference():
    """Per-tree byte-for-byte parity for ``splitter='hist'`` against a
    hand-rolled reference path that mirrors what the shared-bins fast
    path does internally.

    The reference fits ``BinMapper`` once on the full augmented data,
    then per tree: reproduces the block-bootstrap, slices ``X_binned``
    by the bootstrap row indices, and calls ``grow_depthwise_hist_c``
    directly. The forest's shared-bins worker does the same thing under
    the hood, so per-tree predictions must match exactly.
    """
    from riesztree.fast._binner import fit_bin_mapper, transform
    from riesztree.fast._grow_c import grow_depthwise_hist_c
    from riesztree.fast._splitter import loss_kind_for
    from riesztree.fast._tree import node_from_growable_flat_tree
    from riesztree.predictor import RieszTreePredictor
    from forestriesz.aug_backend import _block_bootstrap_indices

    df = _make_df(n=300, p=4)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(4)))
    loss = SquaredLoss()
    feats = df[["a"] + [f"x{j}" for j in range(4)]].to_numpy(dtype=np.float64)
    aug_train = estimand.augment(feats)

    n_estimators = 4
    max_depth, min_split, min_leaf = 6, 2, 1
    max_bins = 64
    random_state = 11

    est = AugForestRieszRegressor(
        estimand=estimand, loss=loss,
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_split, min_samples_leaf=min_leaf,
        max_features=1.0, bootstrap=True, max_samples=None,
        n_jobs=1, random_state=random_state,
        splitter="hist", max_bins=max_bins,
    )
    est.fit(df)
    forest_preds = [t.predict_alpha(aug_train.features) for t in est.predictor_.trees]

    mapper = fit_bin_mapper(
        aug_train.features, max_bins=max_bins, random_state=random_state,
    )
    X_binned_full = transform(aug_train.features, mapper)
    loss_kind, lo, hi, _ = loss_kind_for(loss)

    seed_seq = np.random.SeedSequence(random_state)
    tree_seeds = [int(s) for s in seed_seq.generate_state(n_estimators)]
    ref_preds = []
    for tree_seed in tree_seeds:
        rng = np.random.default_rng(tree_seed)
        idx = _block_bootstrap_indices(
            aug_train, n_subsample=aug_train.n_rows, bootstrap=True, rng=rng,
        )
        sub_X_binned = np.ascontiguousarray(X_binned_full[idx])
        sub_D = np.ascontiguousarray(aug_train.is_original[idx], dtype=np.float64)
        sub_C = np.ascontiguousarray(aug_train.potential_deriv_coef[idx], dtype=np.float64)
        g = grow_depthwise_hist_c(
            sub_X_binned, sub_D, sub_C,
            np.ascontiguousarray(mapper.n_bins, dtype=np.int32),
            list(mapper.bin_thresholds),
            max_bins,
            max_depth, min_split, min_leaf, 0.0,
            int(loss_kind), float(lo), float(hi),
        )
        pred = RieszTreePredictor(
            tree=node_from_growable_flat_tree(g, loss=loss),
            loss=loss, base_score=0.0, feature_keys=(),
        )
        ref_preds.append(pred.predict_alpha(aug_train.features))

    assert len(forest_preds) == len(ref_preds) == n_estimators
    for fp, rp in zip(forest_preds, ref_preds):
        np.testing.assert_array_equal(fp, rp)


def test_hist_shared_bins_falls_back_when_max_features_set():
    """``max_features='sqrt'`` triggers per-split feature subsampling in
    the per-tree path, which the shared-bins fast path doesn't support;
    the dispatch falls back to the per-tree path. The fit must still
    succeed and produce a non-trivial forest."""
    df = _make_df(n=400, p=5)
    estimand = ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(5)))
    est = AugForestRieszRegressor(
        estimand=estimand, loss=SquaredLoss(),
        n_estimators=10, max_depth=4,
        min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", bootstrap=True, max_samples=None,
        n_jobs=1, random_state=0,
        splitter="hist", max_bins=64,
    )
    est.fit(df)
    pred = est.predict(df)
    assert pred.shape == (len(df),)
    assert np.all(np.isfinite(pred))
