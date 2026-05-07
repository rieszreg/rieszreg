# riesztree benchmarks

Wall-clock bench for `RieszTreeRegressor.fit` and `.predict`. Locked-in
baseline numbers on the pure-Python path live in
[`../../BENCH_BASELINE.md`](../../BENCH_BASELINE.md). Every optimisation
phase re-runs `bench_fit.py` and reports its delta vs that baseline.

## Run

```sh
cd /tmp   # cwd must NOT contain a `rieszreg` directory; namespace-package
          # rules would otherwise shadow the editable install
/Users/aschuler/Desktop/RieszReg/.venv/bin/python \
  /Users/aschuler/Desktop/RieszReg/riesztree/python/benchmarks/bench_fit.py \
  --grid small \
  --label my-experiment \
  --out /Users/aschuler/Desktop/RieszReg/riesztree/python/benchmarks/results/my-experiment.csv
```

Grids:

| Grid | Configs | Wall time on pure-Python path |
|---|---|---|
| `quick` | 2 | seconds |
| `small` | 32 | minutes |
| `full`  | 72 | hours+ (target grid for the optimised paths) |

The `small` grid is the locked baseline. The `full` grid is the headline
target — `fit` on `(n=10⁶, p=50, depth=∞)` does not finish on the pure-Python
path but should finish in seconds with `splitter='hist'`.

`--max-fit-seconds` caps any single config's wall time; the run stops
collecting after the first cap-exceeding fit so a bad config doesn't burn the
whole grid.

## What the bench measures

For each `(loss, n, p, max_depth, growth_policy)`:

- `fit_seconds` — wall time of `est.fit(train_df)`.
- `predict_seconds` — wall time of `est.predict(test_df)` on a held-out 10 k rows.
- `n_leaves`, `max_depth_actual` — from `diagnose_tree`.
- `rmse` — RMSE of α̂ vs true Riesz representer on the test set, sanity-check
  that timing comparisons are on equally-fit trees.

## Adding new grids / metrics

Edit `grid_<name>` in `bench_fit.py` and register it in the `GRIDS` dict.
Keep grid names short; `--grid <name>` selects them on the CLI. New metrics
go on the `Result` dataclass and into the CSV columns automatically.
