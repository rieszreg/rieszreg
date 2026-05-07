"""Predictor for the augmentation-style forest backend.

Mirrors ``ForestPredictor`` but trained on augmented evaluation points, so
prediction takes the full feature vector — there is no ``split_feature_indices``
because the splitter saw every feature dimension at fit time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np

from rieszreg import Loss, register_predictor_loader
from rieszreg.losses import loss_from_spec


def _const_phi(features: np.ndarray) -> np.ndarray:
    return np.ones(len(features), dtype=float)


@dataclass
class AugForestPredictor:
    forest: object  # _RieszGRF
    loss: Loss
    base_score: float
    riesz_feature_fns: list[Callable] | None
    # Optional: Bregman-loss leaf overrides keyed by (tree_idx, leaf_node_id).
    # When set, predict_eta uses these instead of the forest's stored leaf
    # values. Always None for SquaredLoss.
    leaf_eta_table: dict[tuple[int, int], np.ndarray] | None = None

    kind: ClassVar[str] = "aug-forestriesz"

    def _phi(self, features: np.ndarray) -> np.ndarray:
        if self.riesz_feature_fns is None:
            return _const_phi(features).reshape(-1, 1)
        return np.column_stack(
            [np.asarray(fn(features), dtype=float) for fn in self.riesz_feature_fns]
        )

    def _theta_from_table(self, features: np.ndarray) -> np.ndarray:
        """Look up the per-tree θ for each test point via the leaf table,
        then average across trees. Returns shape (n, p)."""
        leaves = np.asarray(self.forest.apply(features))   # (n, n_trees)
        n, n_trees = leaves.shape
        p = next(iter(self.leaf_eta_table.values())).shape[0]
        # Per-tree contribution accumulator.
        eta_sum = np.zeros((n, p), dtype=float)
        # Vectorize per-tree by walking unique leaves only.
        for t in range(n_trees):
            leaf_ids_t = leaves[:, t]
            for leaf_id in np.unique(leaf_ids_t):
                key = (int(t), int(leaf_id))
                if key not in self.leaf_eta_table:
                    continue
                mask = leaf_ids_t == leaf_id
                eta_sum[mask] += self.leaf_eta_table[key]
        return eta_sum / float(n_trees)

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        if self.leaf_eta_table is not None:
            theta = self._theta_from_table(features)
        else:
            theta = np.asarray(self.forest.predict(features))
            if theta.ndim == 1:
                theta = theta.reshape(-1, 1)
        phi = self._phi(features)
        return (theta * phi).sum(axis=1) + self.base_score

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        return self.loss.link_to_alpha(self.predict_eta(features))

    def save(self, dir_path) -> None:
        import joblib

        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.forest, path / "forest.joblib")
        extras = {
            "kind": self.kind,
            "loss": self.loss.to_spec(),
            "base_score": float(self.base_score),
            "has_sieve": self.riesz_feature_fns is not None,
            "has_leaf_table": self.leaf_eta_table is not None,
        }
        with open(path / "predictor_extras.json", "w") as f:
            json.dump(extras, f, indent=2)
        if self.leaf_eta_table is not None:
            keys = np.array(
                [[t, l] for (t, l) in self.leaf_eta_table.keys()],
                dtype=np.int64,
            )
            values = np.stack(list(self.leaf_eta_table.values()), axis=0)
            np.savez(path / "leaf_eta_table.npz", keys=keys, values=values)

    @classmethod
    def load(cls, dir_path, *, base_score, loss, best_iteration):
        import joblib

        path = Path(dir_path)
        with open(path / "predictor_extras.json") as f:
            extras = json.load(f)
        forest = joblib.load(path / "forest.joblib")
        leaf_eta_table = None
        if extras.get("has_leaf_table"):
            data = np.load(path / "leaf_eta_table.npz")
            keys = data["keys"]
            values = data["values"]
            leaf_eta_table = {
                (int(k[0]), int(k[1])): values[i] for i, k in enumerate(keys)
            }
        return cls(
            forest=forest,
            loss=loss if loss is not None else loss_from_spec(extras["loss"]),
            base_score=float(extras["base_score"]) if base_score is None else base_score,
            riesz_feature_fns=None,
            leaf_eta_table=leaf_eta_table,
        )


register_predictor_loader("aug-forestriesz", AugForestPredictor.load)
