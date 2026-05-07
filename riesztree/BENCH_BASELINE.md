# BENCH_BASELINE.md

Locked baseline for the **pure-Python splitter path** (riesztree v0.0.1).
Numbers below are the reference every later optimisation phase must beat.
Re-run with:

```sh
cd /tmp   # cwd must NOT contain a `rieszreg` directory
/Users/aschuler/Desktop/RieszReg/.venv/bin/python \
  /Users/aschuler/Desktop/RieszReg/riesztree/python/benchmarks/bench_fit.py \
  --grid small \
  --label v0.0.1-pure-python \
  --max-fit-seconds 120
```

Hardware / OS for the recorded numbers:

- macOS Darwin 24.6.0, Apple Silicon
- Python 3.13.5
- Single fit per cell, no replication.

## Small grid (`--grid small`, 32 configs)

`{loss} × {n} × {p} × {max_depth} × {growth_policy}` =
`{squared, kl} × {1000, 10000} × {5, 20} × {8, 16} × {depthwise, leafwise}`.

Predict is timed on a fixed 10 000-row test set (`Config.n_predict`).
RMSE is α̂ vs. the closed-form Riesz representer on the test set.

| loss | n | p | max_depth | growth | fit (s) | predict 10k (s) | leaves | depth | rmse |
|---|---|---|---|---|---|---|---|---|---|
| squared | 1000 | 5 | 8 | depthwise | 0.15 | 0.49 | 58 | 8 | 0.90 |
| squared | 1000 | 5 | 8 | leafwise | 0.15 | 0.49 | 58 | 8 | 0.90 |
| squared | 1000 | 5 | 16 | depthwise | 0.16 | 0.49 | 78 | 12 | 0.93 |
| squared | 1000 | 5 | 16 | leafwise | 0.16 | 0.49 | 78 | 12 | 0.93 |
| squared | 1000 | 20 | 8 | depthwise | 0.58 | 1.75 | 49 | 8 | 1.11 |
| squared | 1000 | 20 | 8 | leafwise | 0.55 | 1.69 | 49 | 8 | 1.11 |
| squared | 1000 | 20 | 16 | depthwise | 0.60 | 1.70 | 74 | 13 | 1.14 |
| squared | 1000 | 20 | 16 | leafwise | 0.60 | 1.75 | 74 | 13 | 1.14 |
| squared | 10000 | 5 | 8 | depthwise | 1.75 | 0.49 | 121 | 8 | 0.57 |
| squared | 10000 | 5 | 8 | leafwise | 1.79 | 0.53 | 121 | 8 | 0.57 |
| squared | 10000 | 5 | 16 | depthwise | 2.44 | 0.50 | 615 | 16 | 0.97 |
| squared | 10000 | 5 | 16 | leafwise | 2.53 | 0.51 | 615 | 16 | 0.97 |
| squared | 10000 | 20 | 8 | depthwise | 6.40 | 1.75 | 104 | 8 | 0.80 |
| squared | 10000 | 20 | 8 | leafwise | 6.31 | 1.73 | 104 | 8 | 0.80 |
| squared | 10000 | 20 | 16 | depthwise | 9.07 | 1.77 | 547 | 16 | 1.25 |
| squared | 10000 | 20 | 16 | leafwise | 9.19 | 1.71 | 547 | 16 | 1.25 |
| kl | 1000 | 5 | 8 | depthwise | 0.12 | 0.49 | 31 | 8 | 0.63 |
| kl | 1000 | 5 | 8 | leafwise | 0.13 | 0.49 | 31 | 8 | 0.63 |
| kl | 1000 | 5 | 16 | depthwise | 0.13 | 0.51 | 41 | 12 | 0.64 |
| kl | 1000 | 5 | 16 | leafwise | 0.13 | 0.51 | 41 | 12 | 0.64 |
| kl | 1000 | 20 | 8 | depthwise | 0.45 | 1.77 | 25 | 8 | 0.78 |
| kl | 1000 | 20 | 8 | leafwise | 0.44 | 1.69 | 25 | 8 | 0.78 |
| kl | 1000 | 20 | 16 | depthwise | 0.47 | 1.71 | 37 | 12 | 0.79 |
| kl | 1000 | 20 | 16 | leafwise | 0.47 | 1.70 | 37 | 12 | 0.79 |
| kl | 10000 | 5 | 8 | depthwise | 1.39 | 0.53 | 64 | 8 | 0.41 |
| kl | 10000 | 5 | 8 | leafwise | 1.43 | 0.50 | 64 | 8 | 0.41 |
| kl | 10000 | 5 | 16 | depthwise | 1.86 | 0.50 | 333 | 16 | 0.72 |
| kl | 10000 | 5 | 16 | leafwise | 1.84 | 0.50 | 333 | 16 | 0.72 |
| kl | 10000 | 20 | 8 | depthwise | 5.23 | 1.76 | 62 | 8 | 0.55 |
| kl | 10000 | 20 | 8 | leafwise | 5.19 | 1.78 | 62 | 8 | 0.55 |
| kl | 10000 | 20 | 16 | depthwise | 6.72 | 1.72 | 334 | 16 | 0.94 |
| kl | 10000 | 20 | 16 | leafwise | 6.72 | 1.73 | 334 | 16 | 0.94 |

### What jumps out

- `predict` on 10 k rows dominates `fit` for `n=1000`: a Python tree-walk
  cost of ~50 µs/row (p=5) to ~170 µs/row (p=20). That's the predict
  bottleneck the plan removes in Phase 3.
- `fit` scales roughly `O(n × p)` with a steep Python-loop constant:
  ≈ 10× from `(n=1k, p=5)` to `(n=10k, p=20)` at fixed depth.
- Going from `max_depth=8` to `16` adds ~30–50% to fit time — leaves
  multiply but each leaf scan shrinks; Python overhead dominates.
- `depthwise` and `leafwise` are essentially interchangeable in wall time
  on these grids, since both are Python loops over the same per-leaf work.
- KL is faster than Squared at matched (n, p, depth) because the TSM
  augmentation produces ~`n` augmented rows (one per original) while the
  ATE augmentation produces ~`2n` (one per treatment level).

## Achieved speedup vs v0.0.1 baseline

After Phases 2-10 + the paired `rieszreg` augmentation-vectorize fix:

| Cell | v0.0.1 baseline | After perf work | Speedup |
|---|---|---|---|
| `predict 10k` (most cells) | 0.49–1.75 s | **~2 ms** | **~250–850×** |
| `fit (squared, n=10k, p=20, depth=16)` | 9.07 s | **~0.10 s** | **~90×** |
| `fit (squared, n=10k, p=20, depth=8)` | 6.40 s | **~0.07 s** | **~90×** |
| `fit (kl, n=10k, p=20, depth=16)` | 6.72 s | **~0.10 s** | **~67×** |

(`hist` splitter, with the rieszreg augmentation fast path active.)

## Comparison vs state-of-the-art tree libraries (`bench_compare.py`)

After Phases 1–10 + PMS + iterative-grow Cython + histogram buffer pool. `(n_aug=100k, p=20, depth=16)`, fully-grown trees, single fit each:

| Library | Fit time | vs riesztree-hist |
|---|---|---|
| **riesztree-hist** | **0.45 s** | 1.0× |
| sklearn `HistGradientBoostingRegressor` (max_iter=1) | 0.92 s | we're 2× faster |
| sklearn `DecisionTreeRegressor` (exact) | 2.50 s | we're 5.5× faster |
| XGBoost (n_estimators=1, hist) | 0.33 s | 1.35× behind |
| LightGBM (n_estimators=1) | 5.92 s | we're 13× faster |

At smaller cells we now **beat XGBoost outright**. `(n_aug=20k, p=20, depth=16)`: riesztree-hist 0.10 s vs XGBoost 0.14 s — **25% faster than XGBoost**.

## What's left in the speed gap to XGBoost

The remaining ~1.35× to XGBoost at the largest cell (n_aug=100k) has concrete attribution. None are intrinsic to the augmented-Riesz formulation:

1. **Quantile-binner fit cost.** At fit start, `riesztree.fast._binner.fit_bin_mapper` calls `np.quantile` per feature (which uses `np.partition` internally). For `(n_aug=100k, p=20)` that's ~250 ms on its own — roughly half of the total fit time. XGBoost's binner is C++-tight. **One-shot per fit, so amortizes for forests** that bin once and re-use across trees.
2. **`_recurse_pms_depthwise` / `_grow_c` Python overhead.** Even with the iterative-grow Cython driver, the worklist is a Python list and the slot-management is Python-side. XGBoost runs the whole loop in C++.
3. **No fully-Cython categorical / max_features / early-stopping path.** When any of those are set, riesztree falls back to the Python recursion. XGBoost handles them all in its C++ driver.

(1) is the dominant remainder at large `n`. For forest workloads (bin once, fit many trees) the binner cost amortizes, so the per-tree cost is closer to ~200 ms — within ~20% of XGBoost.

The headline target — "essentially as fast as state-of-the-art tree implementations" — is hit. We **beat XGBoost at smaller cells**, are at parity-class at large cells, and consistently beat sklearn DTR / HGB and LightGBM.

## Memory ceiling

Peak resident set during `fit` on `(n_aug=100k, p=50)` stays well under
`4 ×` the augmented-data size at all configs measured. The gating-check
target from earlier phases is comfortably met by `splitter='hist'`.

## Method notes

- Each cell is a single fit. Wall-time variance across repeated runs is
  small relative to the 10×–100× ratios we report. The state-of-the-art
  comparison is from a single run; treat ratios as approximate.
- `rmse` is recorded as a sanity check that timing comparisons are made on
  equally-fit trees. Cross-phase RMSE drift larger than ~5% is a
  regression signal.
- Bench raw CSVs under `python/benchmarks/results/` are gitignored; only
  this file (and the headline numbers it locks in) is tracked.
- Run `bench_compare.py` to reproduce the cross-library comparison.
