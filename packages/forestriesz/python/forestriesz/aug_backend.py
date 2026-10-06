"""AugForestRieszBackend — augmentation-style forest backend.

Implements ``rieszreg.Backend.fit_augmented``. The orchestrator hands this
backend the precomputed ``AugmentedDataset``; the backend fits an ensemble
of ``riesztree.RieszTreeBackend`` instances over block-bootstrapped
subsamples of the augmented rows and averages their per-row predictions.

The backend is fully estimand-agnostic: the augmented row weights
``D_r`` (``is_original``) and ``C_r`` (``potential_deriv_coef``) already
vary across rows for every estimand, so the loss-aware splitter inside
each tree learns from the full feature space without any user-supplied
basis functions.

Hyperparameters mirror :class:`sklearn.ensemble.RandomForestRegressor`
where the augmented Bregman-Riesz setting allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from joblib import Parallel, delayed

from rieszreg import AugmentedDataset, FitResult, Loss
from riesztree import RieszTreeBackend
from riesztree.tree import check_categorical

from .aug_predictor import AugForestPredictor


def _resolve_n_subsample(max_samples: float | int | None, n_rows: int) -> int:
    """Resolve sklearn's ``max_samples`` semantics: ``None`` → full ``n_rows``;
    a float in ``(0, 1]`` → ``round(max_samples * n_rows)``; an int → that count.
    """
    if max_samples is None:
        return n_rows
    if isinstance(max_samples, (int, np.integer)) and not isinstance(
        max_samples, bool
    ):
        if max_samples < 1 or max_samples > n_rows:
            raise ValueError(
                f"max_samples={max_samples} out of range [1, n_rows={n_rows}]."
            )
        return int(max_samples)
    if isinstance(max_samples, float):
        if not (0.0 < max_samples <= 1.0):
            raise ValueError(
                f"max_samples={max_samples} must be in (0.0, 1.0]."
            )
        return max(1, int(round(max_samples * n_rows)))
    raise TypeError(
        f"max_samples must be None, int, or float; got {type(max_samples).__name__}."
    )


def _block_bootstrap_indices(
    aug: AugmentedDataset,
    *,
    n_subsample: int,
    bootstrap: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample original-row indices, then return the matching augmented-row
    indices. Block-level resampling keeps each original row's block of
    augmented rows together; with replacement, a row drawn twice
    contributes its block twice."""
    sampled = rng.choice(aug.n_rows, size=n_subsample, replace=bootstrap)
    if bootstrap:
        counts = np.bincount(sampled, minlength=aug.n_rows)
        return np.repeat(np.arange(aug.features.shape[0]), counts[aug.origin_index])
    keep = np.zeros(aug.n_rows, dtype=bool)
    keep[sampled] = True
    return np.flatnonzero(keep[aug.origin_index])


def _fit_one_tree(
    aug_train: AugmentedDataset,
    X_binned: np.ndarray | None,
    mapper,
    *,
    loss: Loss,
    tree_seed: int,
    n_subsample: int,
    bootstrap: bool,
    tree_backend: RieszTreeBackend,
):
    """Joblib worker: fit one tree on a block-bootstrap subsample. With the
    histogram splitter, ``X_binned`` / ``mapper`` are shared by every tree,
    so each tree slices the pre-binned rows instead of re-binning."""
    idx = _block_bootstrap_indices(
        aug_train, n_subsample=n_subsample, bootstrap=bootstrap,
        rng=np.random.default_rng(tree_seed),
    )
    sub = AugmentedDataset(
        features=aug_train.features[idx],
        is_original=aug_train.is_original[idx],
        potential_deriv_coef=aug_train.potential_deriv_coef[idx],
        origin_index=aug_train.origin_index[idx],
        n_rows=int(n_subsample),
    )
    fit = tree_backend._fit_binned(
        sub, None, loss,
        random_state=tree_seed,
        X_binned=None if X_binned is None else X_binned[idx],
        mapper=mapper,
    )
    return fit.predictor


@dataclass
class AugForestRieszBackend:
    """Augmentation-style random-forest Riesz backend.

    An ensemble of single-tree Riesz regressors fit on the augmented dataset
    of evaluation points with weights ``(D_r, C_r)``. Each tree uses a
    loss-aware splitter that handles every built-in Bregman loss natively.
    No per-estimand configuration is required.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree. ``None`` lets each tree grow until
        leaves saturate ``min_samples_leaf``.
    min_samples_split : int, default=2
        Minimum count of original (D > 0) augmented rows in a node before
        considering a split.
    min_samples_leaf : int, default=1
        Minimum count of original rows in each child of a candidate split.
    min_weight_fraction_leaf : float, default=0.0
        sklearn-parity: leaves must contain at least
        ``ceil(min_weight_fraction_leaf * n_original_total)`` original rows
        (combined with ``min_samples_leaf`` via ``max(...)``).
    max_features : int, float, {"sqrt", "log2"}, or None, default=1.0
        Per-split feature-subsampling rule (sklearn convention).
    max_leaf_nodes : int or None, default=None
        Cap on per-tree leaf count. When set, trees grow best-first
        (as in sklearn); ``None`` grows depth-first with no cap.
    min_impurity_decrease : float, default=0.0
        Reject splits with gain ≤ this threshold.
    ccp_alpha : float, default=0.0
        Cost-complexity pruning penalty applied to each tree. ``0`` disables.
    bootstrap : bool, default=True
        Whether to sample original-row indices with replacement when building
        each tree's training set. ``False`` uses the full set for every tree.
    max_samples : int, float, or None, default=None
        If float in ``(0, 1]``, the per-tree subsample is
        ``round(max_samples * n_rows)`` original rows; if int, that count;
        if ``None``, all ``n_rows`` original rows. Ignored when
        ``bootstrap=False`` and ``max_samples=None`` (every tree sees the
        full set).
    n_jobs : int or None, default=None
        Trees are fit in parallel via :class:`joblib.Parallel`. ``None``
        means one job; ``-1`` means all available cores.
    verbose : int, default=0
        Forwarded to :class:`joblib.Parallel`.
    splitter : {"exact", "hist", "random"}, default="exact"
        Per-tree splitter implementation. ``"exact"`` and ``"hist"`` are
        Cython kernels (the latter using quantile pre-binning into
        ``max_bins`` bins); ``"random"`` draws a single random threshold
        per feature per leaf (sklearn ExtraTrees-style).
    max_bins : int, default=255
        Bin count for the histogram splitter.
    categorical_features : sequence of int or None, default=None
        Column indices (into ``estimand.feature_keys``) treated as integer
        category labels rather than ordered numerics.

    Notes
    -----
    With ``splitter='hist'`` the bin mapper is fitted once on the full
    augmented training data and every tree slices the shared binned matrix
    (the sklearn-HGB convention), instead of re-binning each bootstrap
    subsample.
    """

    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: int | float | str | None = 1.0
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    bootstrap: bool = True
    max_samples: int | float | None = None
    n_jobs: int | None = None
    verbose: int = 0
    splitter: str = "exact"
    max_bins: int = 255
    categorical_features: Sequence[int] | None = None

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: Loss,
        *,
        base_score: float,
        random_state: int | None,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        del hyperparams, base_score  # forest leaves store loss-aware α directly.
        check_categorical(aug_train.features, self.categorical_features or ())

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "max_samples must be None when bootstrap=False (sklearn parity)."
            )
        tree_seeds = [
            int(s) for s in np.random.SeedSequence(random_state).generate_state(self.n_estimators)
        ]
        n_subsample = _resolve_n_subsample(self.max_samples, aug_train.n_rows)

        # sklearn semantics: a leaf cap means best-first growth.
        tree_backend = RieszTreeBackend(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            max_features=self.max_features,
            growth_policy="depthwise" if self.max_leaf_nodes is None else "leafwise",
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            categorical_features=tuple(int(i) for i in self.categorical_features or ()),
            splitter=self.splitter,
            max_bins=self.max_bins,
        )
        X_binned, mapper = tree_backend._bin(aug_train.features, random_state)
        trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_one_tree)(
                aug_train, X_binned, mapper,
                loss=loss,
                tree_seed=seed,
                n_subsample=n_subsample,
                bootstrap=self.bootstrap,
                tree_backend=tree_backend,
            )
            for seed in tree_seeds
        )
        predictor = AugForestPredictor(trees=list(trees), loss=loss)
        has_valid = aug_valid is not None and aug_valid.n_rows > 0
        return FitResult(
            predictor=predictor,
            best_score=(
                aug_valid.mean_loss(loss, predictor.predict_alpha(aug_valid.features))
                if has_valid else None
            ),
        )
