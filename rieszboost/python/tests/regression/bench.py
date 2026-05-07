"""Performance regression benchmark.

Times a fixed reference workload (ATE on the Lee-Schuler binary DGP, n=8000,
200 trees of depth 4) with the default xgboost backend. Writes results to
`bench_results.csv` so trends across commits can be eyeballed.

Not a unit test — this is a script. Run it manually before/after performance-
sensitive changes:

    .venv/bin/python python/tests/regression/bench.py

The numbers are wall-clock and machine-dependent; use them as a within-host
trend, not as an absolute benchmark.
"""

from __future__ import annotations

import csv
import datetime
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rieszboost import ATE, RieszBooster


N = 8_000
N_ESTIMATORS = 200
LEARNING_RATE = 0.05
MAX_DEPTH = 4
SEED = 0


def _df(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    pi = 1 / (1 + np.exp(-(-0.02 * x - x**2 + 4 * np.log(x + 0.3) + 1.5)))
    a = rng.binomial(1, pi)
    return pd.DataFrame({"a": a.astype(float), "x": x})


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def time_fit(n_warmup: int = 1, n_runs: int = 3) -> dict:
    df = _df(N, seed=SEED)
    booster_factory = lambda: RieszBooster(
        estimand=ATE(),
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=0,
    )

    # Warm up (avoid first-import / kernel-cache effects).
    for _ in range(n_warmup):
        booster_factory().fit(df)

    times = []
    for _ in range(n_runs):
        booster = booster_factory()
        t0 = time.perf_counter()
        booster.fit(df)
        times.append(time.perf_counter() - t0)
    return {
        "median_fit_seconds": float(np.median(times)),
        "min_fit_seconds": float(np.min(times)),
        "max_fit_seconds": float(np.max(times)),
        "n_runs": n_runs,
    }


def main():
    print(f"# rieszboost benchmark — n={N}, n_estimators={N_ESTIMATORS}, "
          f"lr={LEARNING_RATE}, depth={MAX_DEPTH}, seed={SEED}")
    stats = time_fit()
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "n": N,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        **stats,
    }
    print(f"  median fit: {stats['median_fit_seconds']:.3f}s "
          f"(min {stats['min_fit_seconds']:.3f}, max {stats['max_fit_seconds']:.3f})")

    csv_path = Path(__file__).parent / "bench_results.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(record)
    print(f"  appended to {csv_path}")


if __name__ == "__main__":
    main()
