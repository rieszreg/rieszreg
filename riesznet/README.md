# riesznet

Neural-network learner for [RieszReg](https://github.com/rieszreg/rieszreg). Trains the Riesz representer α of a linear estimand ψ = E[m(μ)(Z)] using PyTorch.

Sister package to [`rieszboost`](https://github.com/rieszreg/rieszboost) (gradient boosting), [`krrr`](https://github.com/rieszreg/krrr) (kernel ridge), and [`forestriesz`](https://github.com/rieszreg/forestriesz) (random forests). All four packages share the same `rieszreg.RieszEstimator` orchestrator and the same `Estimand` / `LossSpec` family — pick the backend that fits your data and architecture preferences.

Reference: [Chernozhukov, Newey, Quintas-Martínez, Syrgkanis (2021), *RieszNet and ForestRiesz*](https://arxiv.org/abs/2110.03031). Unlike the original RieszNet, this package fits the Riesz representer only (not a joint regression head).

## Install

```sh
pip install -e ./python
```

This package depends on `rieszreg` (the meta-package) and `torch>=2.0`. Clone [rieszreg/rieszreg](https://github.com/rieszreg/rieszreg) as a sibling and `pip install -e ../rieszreg/python` first.

## Quickstart

```python
import numpy as np
import pandas as pd
from riesznet import RieszNet
from rieszreg import ATE

# Toy ATE: A ~ Bernoulli(σ(0.5x)), X ~ N(0,1).
rng = np.random.default_rng(0)
n = 1000
x = rng.normal(size=n)
pi = 1.0 / (1.0 + np.exp(-0.5 * x))
a = (rng.uniform(size=n) < pi).astype(float)
df = pd.DataFrame({"a": a, "x": x})

est = RieszNet(
    estimand=ATE(),
    hidden_sizes=(64, 64),
    epochs=200,
    learning_rate=1e-3,
    batch_size=64,                 # original rows per minibatch; None for full-batch
    early_stopping_rounds=20,
    validation_fraction=0.2,
    random_state=0,
)
est.fit(df)
alpha_hat = est.predict(df)
```

## Custom architectures

The convenience class `RieszNet` covers MLPs. For arbitrary `nn.Module`s, instantiate `TorchBackend` directly and pass it to `rieszreg.RieszEstimator`:

```python
import torch.nn as nn
from rieszreg import RieszEstimator, ATE
from riesznet import TorchBackend

def my_resnet_factory(input_dim):
    return MyResNet(input_dim=input_dim, n_blocks=3, hidden=128)  # outputs scalar eta

def my_optimizer(params):
    return torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-3)

backend = TorchBackend(
    module_factory=my_resnet_factory,
    optimizer_factory=my_optimizer,
    epochs=300,
    batch_size=128,           # in original rows
    device="cuda",
    dtype="float32",
    grad_clip_norm=1.0,
)
est = RieszEstimator(estimand=ATE(), backend=backend)
est.fit(df)
```

`module_factory` must be a top-level (importable) callable so `state_dict` save/load can reconstruct the module by qualname. `functools.partial` over a top-level function is fine.

## R wrapper

Same API surface, MLP knobs only:

```r
library(riesznet)
use_python_riesznet("/path/to/.venv")

est <- RieszNet$new(
  estimand = ATE(treatment = "a", covariates = "x"),
  hidden_sizes = c(64L, 64L),
  epochs = 200L,
  learning_rate = 1e-3,
  early_stopping_rounds = 20L,
  validation_fraction = 0.2,
  random_state = 0L
)
est$fit(df)
alpha_hat <- est$predict(df)
```

Custom torch architectures are Python-only.

## What works today (v0.0.1)

- All six built-in estimands (`ATE`, `ATT`, `TSM`, `AdditiveShift`, `LocalShift`, `StochasticIntervention`) plus custom estimands.
- All four built-in losses (`SquaredLoss`, `KLLoss`, `BernoulliLoss`, `BoundedSquaredLoss`) with autograd-checked gradient parity to the analytic loss spec.
- Architecture flexibility via `TorchBackend(module_factory=..., optimizer_factory=...)`.
- sklearn composition: `clone`, `GridSearchCV`, `cross_val_predict`, `Pipeline`.
- Save / load round-trip with built-in estimands and the default MLP.
- Early stopping by validation Riesz loss.
- CPU, CUDA, and MPS device support.
- R6 wrapper for the simple-MLP path.

## Known sharp edges

- `module_factory` must be importable by qualname for save/load to work. Closures, lambdas, and notebook-cell-defined modules raise on `save()`. Define them at module top level.
- Single-device training only; no multi-GPU or distributed support.
- No mixed precision; `dtype` is `float32` or `float64` end-to-end.
- Bitwise reproducibility on CUDA is not promised; seeding is best-effort.
- The R wrapper exposes simple MLP knobs only. Custom torch architectures are Python-only.

## On the roadmap

- Mixed-precision training for large architectures.
- Optional deterministic-algorithms flag for fully reproducible CUDA runs.
- Reference parity check vs the original RieszNet codebase (head-to-head on a shared DGP).

## License

MIT.
