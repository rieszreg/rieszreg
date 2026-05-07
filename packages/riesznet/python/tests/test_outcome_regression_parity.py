"""OutcomeRegNormSq parity vs. a stock MSE-trained torch MLP.

`m(α)(z, y) = α(x) · y` has Riesz representer μ_0(x) = E[Y | X=x]; under the
squared Bregman-Riesz loss the per-row training objective collapses to
(η - y)², so a Riesz-trained MLP should track an identically-architected
MLP trained with `nn.MSELoss` and the same optimizer.

Pearson > 0.99 is the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from riesznet import RieszNet
from rieszreg import OutcomeRegNormSq, SquaredLoss
from rieszreg.testing.parity import compare


def _regression_df(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    y = np.sin(x0) + 0.5 * x1 + rng.normal(0.0, 0.3, n)
    df = pd.DataFrame({"x0": x0, "x1": x1})
    return df, y


def test_riesznet_outcome_regression_parity_with_mse_mlp():
    df, y = _regression_df(n=200, seed=0)

    common = dict(
        hidden_sizes=(16, 16),
        activation="relu",
        learning_rate=1e-2,
        epochs=200,
        batch_size=None,  # full-batch GD for stable comparison
        random_state=0,
    )

    riesz = RieszNet(
        estimand=OutcomeRegNormSq(covariates=("x0", "x1")),
        loss=SquaredLoss(),
        **common,
    ).fit(df, y)

    # Train an architecturally-matched MLP with plain MSE on (X, y).
    torch.manual_seed(common["random_state"])
    X_t = torch.tensor(df.values, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    layers: list[torch.nn.Module] = []
    prev = X_t.shape[1]
    for h in common["hidden_sizes"]:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ReLU())
        prev = h
    layers.append(torch.nn.Linear(prev, 1))
    ref_net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(ref_net.parameters(), lr=common["learning_rate"])
    for _ in range(common["epochs"]):
        opt.zero_grad()
        loss = ((ref_net(X_t) - y_t) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        ref_pred = ref_net(X_t).numpy().ravel()

    rep = compare(riesz.predict(df), ref_pred)
    assert rep.pearson > 0.99, rep.summary()
