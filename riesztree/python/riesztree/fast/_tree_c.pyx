# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython-compiled predict_alpha_c.

The hot loop walks each row through the flat-array tree at C speed.
Continuous splits run entirely in Cython; categorical splits drop
back to a Python ``in`` check on a ``frozenset`` (the compiled module
holds the GIL on those iterations only). Phase 8 will introduce a
fully-Cython categorical fast path.
"""

import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.float64_t f64
ctypedef cnp.int32_t i32
ctypedef cnp.uint8_t u8


def predict_alpha_c(
    f64[:, ::1] X,
    i32[::1] feature,
    f64[::1] threshold,
    i32[::1] left,
    i32[::1] right,
    u8[::1] is_categorical,
    list cat_left_sets,
    f64[::1] value,
):
    """Per-row tree walk; returns array of α* of length ``X.shape[0]``."""
    cdef Py_ssize_t i
    cdef Py_ssize_t n_rows = X.shape[0]
    cdef Py_ssize_t node
    cdef Py_ssize_t feat
    cdef f64 x_val
    cdef int level
    cdef object cat_set

    out = np.empty(n_rows, dtype=np.float64)
    cdef f64[::1] out_view = out

    for i in range(n_rows):
        node = 0
        while feature[node] >= 0:
            feat = feature[node]
            x_val = X[i, feat]
            if is_categorical[node] != 0:
                # Categorical: fall back to Python frozenset lookup.
                cat_set = cat_left_sets[node]
                level = <int>x_val
                if level in cat_set:
                    node = left[node]
                else:
                    node = right[node]
            else:
                if x_val <= threshold[node]:
                    node = left[node]
                else:
                    node = right[node]
        out_view[i] = value[node]
    return out
