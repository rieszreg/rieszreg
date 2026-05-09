"""Predictor for the augmentation-style forest backend.

Wraps a list of ``riesztree.RieszTreePredictor`` instances and averages
their per-row α predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from rieszreg import Loss, register_predictor_loader
from rieszreg.losses import loss_from_spec
from riesztree import RieszTreePredictor


@dataclass
class AugForestPredictor:
    trees: list[RieszTreePredictor]
    loss: Loss

    kind: ClassVar[str] = "aug-forestriesz"

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        per_tree = np.stack(
            [tree.predict_alpha(features) for tree in self.trees], axis=0
        )
        return per_tree.mean(axis=0)

    def predict_eta(self, features: np.ndarray) -> np.ndarray:
        return self.loss.alpha_to_eta(self.predict_alpha(features))

    def save(self, dir_path) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind,
            "loss": self.loss.to_spec(),
            "n_trees": len(self.trees),
        }
        with open(path / "predictor.json", "w") as f:
            json.dump(meta, f, indent=2)
        for i, tree in enumerate(self.trees):
            tree.save(path / f"tree_{i}")

    @classmethod
    def load(cls, dir_path, *, base_score, loss, best_iteration):
        del base_score, best_iteration
        path = Path(dir_path)
        with open(path / "predictor.json") as f:
            meta = json.load(f)
        resolved_loss = loss if loss is not None else loss_from_spec(meta["loss"])
        n_trees = int(meta["n_trees"])
        trees = [
            RieszTreePredictor.load(
                path / f"tree_{i}",
                base_score=None,
                loss=resolved_loss,
                best_iteration=None,
            )
            for i in range(n_trees)
        ]
        return cls(trees=trees, loss=resolved_loss)


register_predictor_loader("aug-forestriesz", AugForestPredictor.load)
