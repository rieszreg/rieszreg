"""Predictor wrapping the fitted tree.

Holds the tree datastructure plus the loss for link mapping. Squared loss
uses the identity link so ``predict_eta`` and ``predict_alpha`` agree; for
KL / Bernoulli the leaf stores α directly (post-projected to the link
domain), and ``predict_eta`` returns ``loss.alpha_to_eta(α)``.

The predictor stores both the ``Node`` tree (the source of truth used by
diagnostics, pruning, and serialization) and a lazily-built flat-array
companion that backs the Cython ``predict_alpha`` loop. The flat tree
is rebuilt the first time ``predict_alpha`` runs after construction or
load; mutating the ``Node`` tree (e.g. cost-complexity pruning) clears
the cache via :meth:`invalidate_flat_tree`.

Registers itself with rieszreg's loader registry on import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np

from rieszreg import Loss, register_predictor_loader
from rieszreg.losses import loss_from_spec

from .fast import FlatTree, flat_tree_from_node, predict_alpha as _flat_predict_alpha
from .tree import Node, from_dict, to_dict


@dataclass
class RieszTreePredictor:
    tree: Node
    loss: Loss
    base_score: float
    feature_keys: tuple[str, ...]
    categorical_features: tuple[int, ...] = ()
    _flat_tree: FlatTree | None = field(default=None, init=False, repr=False, compare=False)

    kind: ClassVar[str] = "riesztree"

    def _ensure_flat_tree(self) -> FlatTree:
        """Build (and cache) the flat-array companion of ``self.tree``."""
        if self._flat_tree is None:
            self._flat_tree = flat_tree_from_node(self.tree)
        return self._flat_tree

    def invalidate_flat_tree(self) -> None:
        """Drop the cached flat tree; the next ``predict_alpha`` rebuilds it.

        Call after any in-place mutation of ``self.tree`` (e.g. pruning).
        """
        self._flat_tree = None

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        """Per-leaf α* sits in α-space already; return as-is."""
        flat = self._ensure_flat_tree()
        return _flat_predict_alpha(flat, np.asarray(features))

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        alpha = self.predict_alpha(features)
        return self.loss.alpha_to_eta(alpha)

    # ---- serialization ----

    def save(self, dir_path) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
            "feature_keys": list(self.feature_keys),
            "categorical_features": list(int(i) for i in self.categorical_features),
            "tree": to_dict(self.tree),
        }
        with open(path / "predictor.json", "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, dir_path, *, base_score, loss, best_iteration):
        del best_iteration
        path = Path(dir_path)
        with open(path / "predictor.json") as f:
            payload = json.load(f)
        return cls(
            tree=from_dict(payload["tree"]),
            loss=loss if loss is not None else loss_from_spec(payload["loss"]),
            base_score=float(payload["base_score"]) if base_score is None else float(base_score),
            feature_keys=tuple(payload["feature_keys"]),
            categorical_features=tuple(int(i) for i in payload.get("categorical_features", [])),
        )


register_predictor_loader("riesztree", RieszTreePredictor.load)
