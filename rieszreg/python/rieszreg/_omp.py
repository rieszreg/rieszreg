"""Runtime detection of multi-backend OpenMP coexistence."""

from __future__ import annotations

import sys
import warnings


_WARNED = False


def warn_if_multi_backend_omp() -> None:
    """Emit a one-time `RuntimeWarning` if both `torch` and `xgboost` are loaded.

    The condition (both libomp-shipping wheels mapped into one process) is
    necessary but not sufficient for the threadpool deadlock — many users
    who load both never observe a hang. Recovery (set `OMP_NUM_THREADS=1`
    pre-import, or `threadpoolctl.threadpool_limits(1)` around the fit, or
    a conda-forge install) requires the user to act, so the orchestrator
    surfaces the situation rather than silently throttling threads.
    """
    global _WARNED
    if _WARNED:
        return
    if "torch" not in sys.modules or "xgboost" not in sys.modules:
        return
    _WARNED = True
    warnings.warn(
        "Both torch (riesznet) and xgboost (rieszboost) are loaded in this "
        "Python process. macOS pip wheels for these libraries each ship their "
        "own copy of libomp.dylib; when both are mapped into one process, "
        "OMP-parallel work can deadlock non-deterministically. The symptom "
        "is a stuck fit with main-thread frames in __kmp_join_barrier. Linux "
        "pip wheels can similarly throttle threads silently. The hang is not "
        "guaranteed; many users who load both never see it.\n"
        "\n"
        "If you do hit it, restart Python and pick one:\n"
        "  - set OMP_NUM_THREADS=1 (and MKL_NUM_THREADS=1) in the shell or "
        "via os.environ BEFORE importing rieszboost / riesznet / torch / "
        "xgboost (env vars are read at libomp load and ignored after);\n"
        "  - wrap the fit in `with threadpoolctl.threadpool_limits(1):`;\n"
        "  - install via conda-forge, which pins one shared OpenMP runtime.\n"
        "\n"
        "See the troubleshooting page in the user guide for details.",
        RuntimeWarning,
        stacklevel=3,
    )
