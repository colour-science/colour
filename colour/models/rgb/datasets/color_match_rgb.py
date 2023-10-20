"""
ColorMatch RGB Colourspace
==========================

Defines the *ColorMatch RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_COLOR_MATCH_RGB`.

References
----------
-   :cite:`Lindbloom2014a` : Lindbloom, B. (2014). RGB Working Space
    Information. Retrieved April 11, 2014, from
    http://www.brucelindbloom.com/WorkingSpaceInfo.html
"""

from __future__ import annotations

from functools import partial

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
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
    "PRIMARIES_COLOR_MATCH_RGB",
    "WHITEPOINT_NAME_COLOR_MATCH_RGB",
    "CCS_WHITEPOINT_COLOR_MATCH_RGB",
    "MATRIX_COLOR_MATCH_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_COLOR_MATCH_RGB",
    "RGB_COLOURSPACE_COLOR_MATCH_RGB",
]

PRIMARIES_COLOR_MATCH_RGB: NDArrayFloat = np.array(
    [
        [0.6300, 0.3400],
        [0.2950, 0.6050],
        [0.1500, 0.0750],
    ]
)
"""*ColorMatch RGB* colourspace primaries."""

WHITEPOINT_NAME_COLOR_MATCH_RGB: str = "D50"
"""*ColorMatch RGB* colourspace whitepoint name."""

CCS_WHITEPOINT_COLOR_MATCH_RGB: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_COLOR_MATCH_RGB]
"""*ColorMatch RGB* colourspace whitepoint chromaticity coordinates."""

MATRIX_COLOR_MATCH_RGB_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_COLOR_MATCH_RGB, CCS_WHITEPOINT_COLOR_MATCH_RGB
)
"""*ColorMatch RGB* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_COLOR_MATCH_RGB: NDArrayFloat = np.linalg.inv(
    MATRIX_COLOR_MATCH_RGB_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *ColorMatch RGB* colourspace matrix."""

RGB_COLOURSPACE_COLOR_MATCH_RGB: RGB_Colourspace = RGB_Colourspace(
    "ColorMatch RGB",
    PRIMARIES_COLOR_MATCH_RGB,
    CCS_WHITEPOINT_COLOR_MATCH_RGB,
    WHITEPOINT_NAME_COLOR_MATCH_RGB,
    MATRIX_COLOR_MATCH_RGB_TO_XYZ,
    MATRIX_XYZ_TO_COLOR_MATCH_RGB,
    partial(gamma_function, exponent=1 / 1.8),
    partial(gamma_function, exponent=1.8),
)
RGB_COLOURSPACE_COLOR_MATCH_RGB.__doc__ = """
*ColorMatch RGB* colourspace.

References
----------
:cite:`Lindbloom2014a`
"""
