"""Build hook for Cython extensions.

The package metadata lives in ``pyproject.toml``; this file exists
solely so setuptools picks up the Cython-compiled extensions in
``riesztree/fast/``.

To rebuild after editing a ``.pyx`` file::

    pip install -e python/

(``-e .`` from inside ``python/`` works too.)
"""
from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = [
    Extension(
        "riesztree.fast._tree_c",
        sources=["riesztree/fast/_tree_c.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "riesztree.fast._loss_kernels",
        sources=["riesztree/fast/_loss_kernels.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "riesztree.fast._splitter_c",
        sources=["riesztree/fast/_splitter_c.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "riesztree.fast._splitter_hist",
        sources=["riesztree/fast/_splitter_hist.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "riesztree.fast._grow_c",
        sources=["riesztree/fast/_grow_c.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
