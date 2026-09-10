"""
Backend-agnostic constants shared by the interpolator implementations.
"""

from __future__ import annotations

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPRAGUE_C_COEFFICIENTS",
    "SPRAGUE_A_COEFFICIENTS",
]

SPRAGUE_C_COEFFICIENTS: tuple[tuple[float, ...], ...] = (
    (884.0, -1960.0, 3033.0, -2648.0, 1080.0, -180.0),
    (508.0, -540.0, 488.0, -367.0, 144.0, -24.0),
    (-24.0, 144.0, -367.0, 488.0, -540.0, 508.0),
    (-180.0, 1080.0, -2648.0, 3033.0, -1960.0, 884.0),
)
"""*Sprague (1880)* boundary-extension coefficients, shape ``(4, 6)``."""

SPRAGUE_A_COEFFICIENTS: tuple[tuple[float, ...], ...] = (
    (2.0, -16.0, 0.0, 16.0, -2.0, 0.0),
    (-1.0, 16.0, -30.0, 16.0, -1.0, 0.0),
    (-9.0, 39.0, -70.0, 66.0, -33.0, 7.0),
    (13.0, -64.0, 126.0, -124.0, 61.0, -12.0),
    (-5.0, 25.0, -50.0, 50.0, -25.0, 5.0),
)
"""*Sprague (1880)* fifth-order polynomial coefficients, shape ``(5, 6)``."""
