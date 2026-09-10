"""
Array API Interpolation
========================

Backend-specialised, dispatching 1-D interpolators (NumPy / PyTorch / JAX) whose
public API matches ``colour.algebra.interpolation``.
"""

from __future__ import annotations

from ._dispatch import (
    CubicSplineInterpolator,
    Extrapolator,
    KernelInterpolator,
    LinearInterpolator,
    NearestNeighbourInterpolator,
    NullInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
    detect_backend,
)
from .backends._kernels import (
    kernel_cardinal_spline,
    kernel_lanczos,
    kernel_linear,
    kernel_nearest_neighbour,
    kernel_sinc,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CubicSplineInterpolator",
    "LinearInterpolator",
    "NearestNeighbourInterpolator",
    "NullInterpolator",
    "PchipInterpolator",
    "SpragueInterpolator",
    "detect_backend",
    "Extrapolator",
    "KernelInterpolator",
    "kernel_cardinal_spline",
    "kernel_lanczos",
    "kernel_linear",
    "kernel_nearest_neighbour",
    "kernel_sinc",
]
