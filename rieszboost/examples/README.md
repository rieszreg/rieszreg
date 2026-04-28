# Examples

End-to-end scripts demonstrating `rieszboost` and reproducing published results.

## Lee & Schuler (arXiv:2501.04871)

Two scripts reproduce Section 4 of the paper. Each script simulates one of the paper's data-generating processes, fits an outcome regression `μ̂` and a Riesz representer `α̂`, and reports both the Riesz-RMSE and the final-parameter EEE estimate with coverage.

```sh
# Section 4.1: binary treatment, ATE + ATT
.venv/bin/python examples/lee_schuler/binary_dgp.py --n_reps 50

# Section 4.2: continuous treatment, ASE + LASE
.venv/bin/python examples/lee_schuler/continuous_dgp.py --n_reps 50
```

Defaults: `n=1000` per rep (500 train / 500 estimation), `n_reps=50`. Pass `--n_reps 500` to match the paper's full study (~30–60 min wall time).

### What gets reproduced

The paper reports:

- **Table 1** — Riesz-representer estimation RMSE/MAE for ATE/ATT under the binary DGP.
- **Tables 2 & 3** — final EEE estimates of ATE / ATT (mean, average SE, RMSE, empirical SD, 95% coverage).
- **Table 4** — Riesz-representer RMSE/MAE for ASE/LASE under the continuous DGP.
- **Tables 5 & 6** — final EEE estimates of ASE / LASE.

Both scripts print the paper's reported values alongside ours for direct comparison.

### Notes on hyperparameters

Lee-Schuler tune over a grid (`learning_rate ∈ {0.001, 0.01, 0.1, 0.25}`, `max_depth ∈ {3, 5, 7}`, `M ∈ {10, 30, …, 200}`) via cross-validation. The example scripts use a single fixed setting (`learning_rate=0.01, max_depth=3, reg_lambda=1`) plus early stopping on a held-out 20% of the training set. Final-parameter estimates are typically within a fraction of an SE of the paper; α RMSE is somewhat higher without grid tuning. Plug a CV loop around the `fit(...)` call to close the remaining gap.

### Departures from the paper

- The ATT example uses Lee-Schuler's "partial-parameter" formulation `m(O, μ) = A(μ(1, X) − μ(0, X))` with the EEE delta-method correction for `1/P(A=1)`. Built-in `rieszboost.ATT(p_treated)` uses the equivalent direct formulation `m(O, μ) = (A/p_treated)(μ(1, X) − μ(0, X))`; either works.
- Same for LASE, which uses the partial-parameter form `m(O, μ) = 1(A < t)(μ(A+δ, X) − μ(A, X))` followed by the delta-method correction for `1/P(A < t)`.

### Cross-check vs the Lee-Schuler reference implementation

A standalone comparison script verifies that our `gradient_only=True` engine reproduces Lee-Schuler's reference implementation ([`kaitlynjlee/boosting_for_rr`](https://github.com/kaitlynjlee/boosting_for_rr)). See [COMPARISON.md](lee_schuler/COMPARISON.md) for the full writeup, including the math walkthrough behind the `learning_rate/2` rescaling and a bug we caught in the reference's no-early-stopping path. Headline:

| | Pearson(ref, ours) | RMSE(ref vs ours) | RMSE vs truth |
|---|---|---|---|
| ATE | 0.998 | 0.13 | ~1.0 |
| ATT | 0.986 | 0.19 | ~0.75 |

### Caveat — LASE

LASE's true Riesz representer has step discontinuities at `A = 0` and `A = 1` (the indicator boundaries). Trees smooth these into ramps, biasing α̂ near the boundaries and hence the EEE estimate. Lee-Schuler's CV-tuned configuration handles this better than our fixed hyperparameters; with a CV wrapper (or much larger n), our LASE coverage approaches theirs.
