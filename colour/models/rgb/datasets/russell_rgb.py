"""
Russell RGB Colourspace
=======================

Defines the *Russell RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_RUSSELL_RGB`.

References
----------
-   :cite:`Cottrella` : Cottrell, R. (n.d.). The Russell RGB working color
    space. http://www.russellcottrell.com/photo/downloads/RussellRGB.icc
"""

from __future__ import annotations

from functools import partial

import numpy as np

from colour.colorimetry.datasets import CCS_ILLUMINANTS
from colour.hints import NDArrayFloat
from colour.models.rgb import (
    RGB_Colourspace,
    gamma_function,
    normalised_primary_matrix,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_RUSSELL_RGB",
    "WHITEPOINT_NAME_RUSSELL_RGB",
    "CCS_WHITEPOINT_RUSSELL_RGB",
    "MATRIX_RUSSELL_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_RUSSELL_RGB",
    "RGB_COLOURSPACE_RUSSELL_RGB",
]

PRIMARIES_RUSSELL_RGB: NDArrayFloat = np.array(
    [
        [0.6900, 0.3100],
        [0.1800, 0.7700],
        [0.1000, 0.0200],
    ]
)
"""*Russell RGB* colourspace primaries."""

WHITEPOINT_NAME_RUSSELL_RGB: str = "D55"
"""*Russell RGB* colourspace whitepoint name."""

CCS_WHITEPOINT_RUSSELL_RGB: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_RUSSELL_RGB]
"""*Russell RGB* colourspace whitepoint chromaticity coordinates."""

MATRIX_RUSSELL_RGB_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_RUSSELL_RGB, CCS_WHITEPOINT_RUSSELL_RGB
)
"""*Russell RGB* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_RUSSELL_RGB: NDArrayFloat = np.linalg.inv(
    MATRIX_RUSSELL_RGB_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *Russell RGB* colourspace matrix."""

RGB_COLOURSPACE_RUSSELL_RGB: RGB_Colourspace = RGB_Colourspace(
    "Russell RGB",
    PRIMARIES_RUSSELL_RGB,
    CCS_WHITEPOINT_RUSSELL_RGB,
    WHITEPOINT_NAME_RUSSELL_RGB,
    MATRIX_RUSSELL_RGB_TO_XYZ,
    MATRIX_XYZ_TO_RUSSELL_RGB,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_RUSSELL_RGB.__doc__ = """
*Russell RGB* colourspace.

References
----------
:cite:`Cottrella`
"""
