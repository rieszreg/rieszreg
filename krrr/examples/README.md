# krrr examples

One worked script per built-in estimand. Each runs end-to-end from a synthetic
DGP to a fitted Riesz representer + diagnostics output. Run from the repo
root, e.g.:

```sh
.venv/bin/python examples/ate_quickstart.py
```

| Script | Estimand | DGP |
|---|---|---|
| `ate_quickstart.py` | `ATE` | binary treatment, logistic propensity |
| `att_quickstart.py` | `ATT` (partial) | binary treatment, logistic propensity |
| `tsm_quickstart.py` | `TSM(level=1)` | binary treatment, logistic propensity |
| `additive_shift_quickstart.py` | `AdditiveShift(δ=0.2)` | continuous treatment |
| `local_shift_quickstart.py` | `LocalShift(δ=0.2, threshold=0.5)` (partial) | continuous treatment |
| `stochastic_intervention_quickstart.py` | `StochasticIntervention(samples_key="shift_samples")` | continuous treatment, MTP |

**Reference parity** vs the [`dml-tmle`](https://github.com/alejandroschuler/dml-tmle) `krrr.R`
reference implementation lives in the test suite at
[`python/tests/test_reference_parity.py`](../python/tests/test_reference_parity.py),
which reproduces TSM1 predictions to tolerance 1e-8.
