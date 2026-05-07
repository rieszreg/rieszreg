# cython: language_level=3
"""Cython header file declaring leaf-loss + alpha-at-opt kernels.

Other compiled modules ``cimport`` the dispatcher to call the kernels at
C-call speed (no Python overhead per evaluation). The integer ``loss_kind``
parameter is the public selector; the four built-in losses map to the
``LOSS_*`` constants below.
"""

cdef int LOSS_SQUARED
cdef int LOSS_KL
cdef int LOSS_BERNOULLI
cdef int LOSS_BOUNDED_SQUARED


cdef double dispatch_leaf_loss(
    int loss_kind, double D, double C, double bounded_lo, double bounded_hi
) noexcept nogil


cdef double dispatch_alpha_at_opt(
    int loss_kind, double D, double C, double bounded_lo, double bounded_hi
) noexcept nogil
