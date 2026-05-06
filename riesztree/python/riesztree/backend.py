"""RieszTreeBackend — augmentation-style single-tree backend.

Implements ``rieszreg.Backend.fit_augmented``. Consumes the precomputed
``AugmentedDataset``, picks the loss-aware splitter, and grows / prunes
the tree according to the constructor's hyperparameters.

Universal across the four built-in losses (SquaredLoss / KLLoss /
BernoulliLoss / BoundedSquaredLoss). Custom Loss subclasses raise
NotImplementedError from the leaf-solver dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from rieszreg import (
    AugmentedDataset,
    FitResult,
    LossSpec,
    SquaredLoss,
    aug_loss_alpha,
)

from .grow import grow_depthwise, grow_leafwise
from .predictor import RieszTreePredictor
from .pruning import cost_complexity_prune
from .splitter import make_leaf_solvers
from .tree import Node


def _holdout_riesz_loss(
    tree: Node, aug_valid: AugmentedDataset, loss: LossSpec
) -> float:
    from .tree import predict_array

    alpha_hat = predict_array(tree, aug_valid.features)
    return float(
        np.sum(aug_loss_alpha(
            loss, aug_valid.is_original, aug_valid.potential_deriv_coef, alpha_hat
        ))
        / aug_valid.n_rows
    )


@dataclass
class RieszTreeBackend:
    """Single-tree backend for the augmentation-style entry point.

    Hyperparameters mirror :class:`sklearn.tree.DecisionTreeRegressor`
    where the augmented Bregman-Riesz setting allows.

    Parameters
    ----------
    max_depth
        Maximum tree depth. Default 8.
    min_samples_split
        Minimum count of original (D > 0) augmented rows in a node before
        considering a split. Default 20.
    min_samples_leaf
        Minimum count of original rows in each child of a candidate split.
        Default 10.
    min_weight_fraction_leaf
        sklearn-parity: leaves must contain at least
        ``ceil(min_weight_fraction_leaf * n_original_total)`` original rows
        (combined with ``min_samples_leaf`` via ``max(...)``). Default 0.0.
    max_leaf_nodes
        Cap for leafwise growth. Ignored when ``growth_policy="depthwise"``.
        Default 31.
    max_features
        Per-split feature-subsampling rule, as in
        :class:`sklearn.tree.DecisionTreeRegressor`. ``None``, ``"sqrt"``,
        ``"log2"``, an int, or a float in ``(0, 1]``. Default ``None``.
    growth_policy
        ``"depthwise"`` (recursive depth-first) or ``"leafwise"`` (best-first).
        Default ``"depthwise"``.
    min_impurity_decrease
        Reject splits with gain ≤ this threshold. Default 0.0 (sklearn).
    ccp_alpha
        Cost-complexity penalty. ``0`` (default) disables pruning.
    early_stopping_rounds
        Stop growing when held-out augmented loss has not improved for that
        many consecutive accepted splits. ``None`` (default) disables.
    validation_fraction
        Held-out fraction the orchestrator splits off before augmentation
        when early stopping or pruning needs a holdout. Default 0.0.
    categorical_features
        Tuple of column indices (into the estimand's ``feature_keys``)
        whose values should be treated as integer category labels rather
        than ordered numerics. Default ``()``.
    random_state
        Seeds the per-split feature subsample under ``max_features``.
        Default 0.
    splitter
        ``"exact"`` (default) routes continuous-feature splits through
        the Cython sweep in :mod:`riesztree.fast._splitter_c`;
        ``"hist"`` uses the histogram-based Cython splitter
        (:mod:`riesztree.fast._splitter_hist`) with quantile pre-binning;
        ``"random"`` (sklearn ExtraTrees-style) draws a single uniform
        threshold per feature per leaf and evaluates the gain there;
        ``"python"`` keeps the original pure-Python splitter (kept for
        debugging and as the fallback for losses outside the four
        built-ins).
    """

    max_depth: int = 8
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    min_weight_fraction_leaf: float = 0.0
    max_leaf_nodes: int = 31
    max_features: object = None     # int | float | str | None
    growth_policy: str = "depthwise"
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    early_stopping_rounds: int | None = None
    validation_fraction: float = 0.0
    categorical_features: tuple[int, ...] = field(default_factory=tuple)
    random_state: int = 0
    splitter: str = "exact"
    max_bins: int = 255      # used when splitter == "hist"

    def fit_augmented(
        self,
        aug_train: AugmentedDataset,
        aug_valid: AugmentedDataset | None,
        loss: LossSpec,
        *,
        base_score: float,
        random_state: int,
        hyperparams: dict[str, Any],
    ) -> FitResult:
        del hyperparams, random_state, base_score  # tree leaves are loss-aware; no boosting offset.

        if self.growth_policy not in ("depthwise", "leafwise"):
            raise ValueError(
                f"growth_policy must be 'depthwise' or 'leafwise'; got "
                f"{self.growth_policy!r}."
            )

        # The leaf solver dispatcher raises a clear NotImplementedError for
        # unsupported Loss subclasses; fail fast.
        make_leaf_solvers(loss)

        valid_tuple = None
        if aug_valid is not None and aug_valid.n_rows > 0:
            valid_tuple = (
                aug_valid.features,
                aug_valid.is_original,
                aug_valid.potential_deriv_coef,
            )

        cat_feats = tuple(int(i) for i in self.categorical_features)

        # Histogram pre-binning (one-shot at fit start). Categorical
        # columns are excluded from binning — they go through the
        # Python categorical splitter regardless.
        hist_payload = None
        if self.splitter == "hist":
            from .fast._binner import fit_bin_mapper, transform
            mapper = fit_bin_mapper(
                aug_train.features,
                max_bins=self.max_bins,
                random_state=self.random_state,
            )
            X_binned = transform(aug_train.features, mapper)
            hist_payload = {
                "X_binned": X_binned,
                "bin_thresholds": mapper.bin_thresholds,
                "n_bins_per_feature": mapper.n_bins,
                "max_bins": int(self.max_bins),
            }

        if self.growth_policy == "depthwise":
            tree = grow_depthwise(
                features=aug_train.features,
                D=aug_train.is_original,
                C=aug_train.potential_deriv_coef,
                loss=loss,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_orig_leaf=self.min_samples_leaf,
                categorical_features=cat_feats,
                aug_valid=valid_tuple,
                early_stopping_rounds=self.early_stopping_rounds,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                random_state=self.random_state,
                splitter=self.splitter,
                hist_payload=hist_payload,
            )
        else:
            tree = grow_leafwise(
                features=aug_train.features,
                D=aug_train.is_original,
                C=aug_train.potential_deriv_coef,
                loss=loss,
                max_leaf_nodes=self.max_leaf_nodes,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_orig_leaf=self.min_samples_leaf,
                categorical_features=cat_feats,
                aug_valid=valid_tuple,
                early_stopping_rounds=self.early_stopping_rounds,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                random_state=self.random_state,
                splitter=self.splitter,
                hist_payload=hist_payload,
            )

        if self.ccp_alpha > 0:
            tree = cost_complexity_prune(tree, loss, ccp_alpha=self.ccp_alpha)

        # The orchestrator uses ``base_score`` to seed boosting / additive
        # learners; for a tree the leaves already store the loss-aware α*.
        # We hand the predictor base_score=0.0 so predict_eta = link^{-1}(α).
        feature_keys = tuple()  # patched by the convenience class via attribute
        predictor = RieszTreePredictor(
            tree=tree,
            loss=loss,
            base_score=0.0,
            feature_keys=feature_keys,
            categorical_features=cat_feats,
        )

        val_score = None
        if aug_valid is not None and aug_valid.n_rows > 0:
            val_score = _holdout_riesz_loss(tree, aug_valid, loss)

        return FitResult(
            predictor=predictor,
            best_iteration=None,
            best_score=val_score,
            history=None,
        )
