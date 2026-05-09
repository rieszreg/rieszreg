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

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from joblib import Parallel, delayed

from rieszreg import (
    AugmentedDataset,
    FitResult,
    Loss,
    SquaredLoss,
    KLLoss,
    BernoulliLoss,
    BoundedSquaredLoss,
)
from riesztree import RieszTreeBackend

from .aug_predictor import AugForestPredictor


def _resolve_n_subsample(
    max_samples: float | int | None, n_rows: int, bootstrap: bool
) -> int:
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


def _block_bootstrap(
    aug: AugmentedDataset,
    *,
    n_subsample: int,
    bootstrap: bool,
    rng: np.random.Generator,
) -> AugmentedDataset:
    """Sample original-row indices, then expand to the matching augmented rows.

    Block-level resampling preserves the per-original block of correlated
    augmented rows that ``Estimand.augment`` produces.
    """
    sampled = rng.choice(aug.n_rows, size=n_subsample, replace=bootstrap)
    # Build a mask over augmented rows: include row r iff origin_index[r]
    # appears in `sampled`. With replacement, a duplicated original index
    # contributes its augmented block multiple times.
    if bootstrap:
        counts = np.bincount(sampled, minlength=aug.n_rows)
        repeats = counts[aug.origin_index]
        idx = np.repeat(np.arange(aug.features.shape[0]), repeats)
    else:
        keep = np.zeros(aug.n_rows, dtype=bool)
        keep[sampled] = True
        idx = np.flatnonzero(keep[aug.origin_index])
    return AugmentedDataset(
        features=aug.features[idx],
        is_original=aug.is_original[idx],
        potential_deriv_coef=aug.potential_deriv_coef[idx],
        origin_index=aug.origin_index[idx],
        n_rows=int(n_subsample),
    )


def _block_bootstrap_indices(
    aug: AugmentedDataset,
    *,
    n_subsample: int,
    bootstrap: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Same draw as :func:`_block_bootstrap` but returns the augmented-row
    index array directly, so the caller can apply it to a pre-binned
    ``X_binned`` view in addition to the raw features."""
    sampled = rng.choice(aug.n_rows, size=n_subsample, replace=bootstrap)
    if bootstrap:
        counts = np.bincount(sampled, minlength=aug.n_rows)
        repeats = counts[aug.origin_index]
        return np.repeat(np.arange(aug.features.shape[0]), repeats)
    keep = np.zeros(aug.n_rows, dtype=bool)
    keep[sampled] = True
    return np.flatnonzero(keep[aug.origin_index])


def _fit_one_tree(
    aug_train: AugmentedDataset,
    *,
    loss: Loss,
    tree_seed: int,
    n_subsample: int,
    bootstrap: bool,
    backend_kwargs: dict[str, Any],
) -> Any:
    """Worker — runs inside ``joblib.Parallel``. Returns the tree's predictor."""
    rng = np.random.default_rng(tree_seed)
    sub = _block_bootstrap(
        aug_train, n_subsample=n_subsample, bootstrap=bootstrap, rng=rng
    )
    tree_backend = RieszTreeBackend(random_state=tree_seed, **backend_kwargs)
    fit = tree_backend.fit_augmented(
        sub, None, loss, base_score=0.0, random_state=tree_seed, hyperparams={}
    )
    return fit.predictor


def _fit_one_tree_hist_shared_bins(
    aug_train: AugmentedDataset,
    X_binned_full: np.ndarray,
    bin_thresholds: list,
    n_bins_per_feature: np.ndarray,
    *,
    loss: Loss,
    tree_seed: int,
    n_subsample: int,
    bootstrap: bool,
    max_depth: int,
    min_samples_split: int,
    min_orig_leaf: int,
    min_impurity_decrease: float,
    max_bins: int,
    feature_keys: tuple,
    loss_kind: int,
    bounded_lo: float,
    bounded_hi: float,
) -> Any:
    """Joblib worker: fit one tree on a bootstrap subsample using a
    SHARED pre-binned feature matrix and bin thresholds.

    Skips the per-tree :func:`fit_bin_mapper` + :func:`transform` work
    that :class:`RieszTreeBackend.fit_augmented` would otherwise repeat
    on every bootstrap subsample, and calls
    :func:`riesztree.fast._grow_c.grow_depthwise_hist_c` directly with
    the shared thresholds. Algorithmically equivalent to fitting the
    bin mapper on the full augmented data and reusing it across trees
    (sklearn-HGB convention).
    """
    from riesztree.fast._grow_c import grow_depthwise_hist_c
    from riesztree.fast._tree import node_from_growable_flat_tree
    from riesztree.predictor import RieszTreePredictor

    rng = np.random.default_rng(tree_seed)
    idx = _block_bootstrap_indices(
        aug_train, n_subsample=n_subsample, bootstrap=bootstrap, rng=rng,
    )
    if idx.size == 0:
        # Pathological draw: empty subsample. Build a degenerate root-only
        # leaf at α=0.
        from riesztree.tree import Node
        from riesztree.fast._loss_kernels import py_dispatch_alpha_at_opt
        root_alpha = py_dispatch_alpha_at_opt(int(loss_kind), 0.0, 0.0, bounded_lo, bounded_hi)
        node = Node(is_leaf=True, alpha=float(root_alpha), depth=0)
        return RieszTreePredictor(
            tree=node, loss=loss, base_score=0.0, feature_keys=feature_keys,
            categorical_features=(),
        )

    sub_X_binned = np.ascontiguousarray(X_binned_full[idx])
    sub_D = np.ascontiguousarray(aug_train.is_original[idx], dtype=np.float64)
    sub_C = np.ascontiguousarray(aug_train.potential_deriv_coef[idx], dtype=np.float64)
    growable = grow_depthwise_hist_c(
        sub_X_binned, sub_D, sub_C,
        np.ascontiguousarray(n_bins_per_feature, dtype=np.int32),
        list(bin_thresholds),
        int(max_bins),
        int(max_depth),
        int(min_samples_split),
        int(min_orig_leaf),
        float(min_impurity_decrease),
        int(loss_kind),
        float(bounded_lo),
        float(bounded_hi),
    )
    return RieszTreePredictor(
        tree=node_from_growable_flat_tree(growable, loss=loss),
        loss=loss, base_score=0.0, feature_keys=feature_keys,
        categorical_features=(),
    )


def _holdout_riesz_loss(
    aug_valid: AugmentedDataset, predictor: AugForestPredictor, loss: Loss
) -> float:
    alpha_hat = predictor.predict_alpha(aug_valid.features)
    return float(
        np.sum(
            loss.aug_loss_alpha(
                aug_valid.is_original, aug_valid.potential_deriv_coef, alpha_hat
            )
        )
        / aug_valid.n_rows
    )


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
        Cap on per-tree leaf count. ``None`` disables the cap.
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
    random_state : int, default=0
        Seeds the per-tree random states (block-bootstrap and per-split
        feature subsample).
    verbose : int, default=0
        Forwarded to :class:`joblib.Parallel`.
    splitter : {"exact", "hist", "random", "python"}, default="exact"
        Per-tree splitter implementation. ``"exact"`` and ``"hist"`` are
        Cython kernels (the latter using quantile pre-binning into
        ``max_bins`` bins); ``"random"`` draws a single random threshold
        per feature per leaf (sklearn ExtraTrees-style); ``"python"`` is
        the slow reference implementation.
    max_bins : int, default=255
        Bin count for the histogram splitter.
    categorical_features : sequence of int or None, default=None
        Column indices (into ``estimand.feature_keys``) treated as integer
        category labels rather than ordered numerics.

    Notes
    -----
    When ``splitter='hist'`` and the configuration has no categoricals,
    no per-split ``max_features`` subsampling, no cost-complexity pruning,
    and no leafwise / max_leaf_nodes cap, the bin mapper is fitted **once**
    on the full augmented training data and the binned matrix is shared
    across joblib workers — algorithmically identical to fitting the
    mapper on each bootstrap subsample at our typical sample sizes
    (sklearn-HGB convention), but avoids re-binning ``n_estimators`` times.
    The win is largest at shallow depths where per-tree binning dominates
    tree-build wall time.
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
    random_state: int = 0
    verbose: int = 0
    splitter: str = "exact"
    max_bins: int = 255
    categorical_features: Sequence[int] | None = None

    def _hist_shared_bins_eligible(self, loss: Loss) -> bool:
        """Whether the hist-mode shared-bin-mapper fast path applies.

        Conservative criteria — anything outside this set falls back to
        the per-tree path (which re-fits the bin mapper per bootstrap).
        """
        if self.splitter != "hist":
            return False
        if self.categorical_features:
            return False
        mf = self.max_features
        if mf is not None and mf != "all" and mf != 1.0:
            return False
        if self.ccp_alpha and float(self.ccp_alpha) > 0.0:
            return False
        if self.max_leaf_nodes is not None and int(self.max_leaf_nodes) < (2 ** 31 - 1):
            return False
        if not isinstance(loss, (SquaredLoss, KLLoss, BernoulliLoss, BoundedSquaredLoss)):
            return False
        return True

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: Loss,
        *,
        base_score: float,
        random_state: int,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        del hyperparams, base_score  # forest leaves store loss-aware α directly.

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "max_samples must be None when bootstrap=False (sklearn parity)."
            )

        seed = random_state if random_state is not None else self.random_state
        seed_seq = np.random.SeedSequence(seed)
        tree_seeds = [int(s) for s in seed_seq.generate_state(self.n_estimators)]

        n_subsample = (
            aug_train.n_rows
            if self.max_samples is None
            else _resolve_n_subsample(
                self.max_samples, aug_train.n_rows, self.bootstrap
            )
        )

        if self._hist_shared_bins_eligible(loss):
            # Pre-bin once on the full augmented training data and reuse
            # the bin mapper across every tree. Each joblib worker
            # subsamples ``X_binned`` and runs ``grow_depthwise_hist_c``
            # directly. Skips the per-tree ``fit_bin_mapper + transform``
            # work that ``RieszTreeBackend.fit_augmented`` would do
            # otherwise.
            from riesztree.fast._binner import fit_bin_mapper, transform
            from riesztree.fast._splitter import loss_kind_for

            mapper = fit_bin_mapper(
                aug_train.features, max_bins=int(self.max_bins),
                random_state=int(self.random_state),
            )
            X_binned = transform(aug_train.features, mapper)
            n_bins_per_feature = np.ascontiguousarray(mapper.n_bins, dtype=np.int32)
            bin_thresholds = list(mapper.bin_thresholds)
            loss_kind, bounded_lo, bounded_hi, _ = loss_kind_for(loss)
            eff_max_depth = (2 ** 31 - 1) if self.max_depth is None else int(self.max_depth)
            # ``min_samples_leaf`` is the per-child original-row count. The
            # ``min_weight_fraction_leaf`` knob would tighten it, but the
            # eligibility check rejects non-default values via the
            # per-tree-path branch when needed.
            import math
            n_orig_total = int(aug_train.n_rows)
            mwfl = float(self.min_weight_fraction_leaf)
            if mwfl <= 0.0:
                eff_min_orig_leaf = int(self.min_samples_leaf)
            else:
                eff_min_orig_leaf = int(max(
                    self.min_samples_leaf,
                    math.ceil(mwfl * max(n_orig_total, 0)),
                ))

            tree_predictors = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_fit_one_tree_hist_shared_bins)(
                    aug_train, X_binned, bin_thresholds, n_bins_per_feature,
                    loss=loss,
                    tree_seed=tree_seeds[i],
                    n_subsample=n_subsample,
                    bootstrap=self.bootstrap,
                    max_depth=eff_max_depth,
                    min_samples_split=int(self.min_samples_split),
                    min_orig_leaf=eff_min_orig_leaf,
                    min_impurity_decrease=float(self.min_impurity_decrease),
                    max_bins=int(self.max_bins),
                    feature_keys=(),
                    loss_kind=int(loss_kind),
                    bounded_lo=float(bounded_lo),
                    bounded_hi=float(bounded_hi),
                )
                for i in range(self.n_estimators)
            )
        else:
            cat = (
                tuple(int(i) for i in self.categorical_features)
                if self.categorical_features is not None
                else ()
            )
            # Riesztree's leafwise growth ignores max_leaf_nodes only when
            # max_leaf_nodes is set; for forests we default to depthwise growth
            # and pass a large cap so it doesn't bind unless the user set it.
            tree_kwargs = dict(
                max_depth=(2**31 - 1) if self.max_depth is None else int(self.max_depth),
                min_samples_split=int(self.min_samples_split),
                min_samples_leaf=int(self.min_samples_leaf),
                min_weight_fraction_leaf=float(self.min_weight_fraction_leaf),
                max_leaf_nodes=(
                    (2**31 - 1) if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
                ),
                max_features=self.max_features,
                growth_policy="depthwise",
                min_impurity_decrease=float(self.min_impurity_decrease),
                ccp_alpha=float(self.ccp_alpha),
                early_stopping_rounds=None,
                validation_fraction=0.0,
                categorical_features=cat,
                splitter=self.splitter,
                max_bins=int(self.max_bins),
            )

            tree_predictors = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_fit_one_tree)(
                    aug_train,
                    loss=loss,
                    tree_seed=tree_seeds[i],
                    n_subsample=n_subsample,
                    bootstrap=self.bootstrap,
                    backend_kwargs=tree_kwargs,
                )
                for i in range(self.n_estimators)
            )

        predictor = AugForestPredictor(
            trees=list(tree_predictors),
            loss=loss,
        )

        val_score = None
        if aug_valid is not None and aug_valid.n_rows > 0:
            val_score = _holdout_riesz_loss(aug_valid, predictor, loss)

        return FitResult(
            predictor=predictor,
            best_iteration=None,
            best_score=val_score,
            history=None,
        )
