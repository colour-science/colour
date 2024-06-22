"""
Adobe Wide Gamut RGB Colourspace
================================

Define the *Adobe Wide Gamut RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB`.

References
----------
-   :cite:`Wikipedia2004c` : Wikipedia. (2004). Wide-gamut RGB color space.
    Retrieved April 13, 2014, from
    http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArrayFloat
from colour.models.rgb import RGB_Colourspace, normalised_primary_matrix
from colour.models.rgb.transfer_functions.gamma import (
    decoding_gamma_function_2_199,
    encoding_gamma_function_2_199,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_ADOBE_WIDE_GAMUT_RGB",
    "WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB",
    "CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB",
    "MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_ADOBE_WIDE_GAMUT_RGB",
    "RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB",
]

PRIMARIES_ADOBE_WIDE_GAMUT_RGB: NDArrayFloat = np.array(
    [
        [0.7347, 0.2653],
        [0.1152, 0.8264],
        [0.1566, 0.0177],
    ]
)
"""*Adobe Wide Gamut RGB* colourspace primaries."""

WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB: str = "D50"
"""*Adobe Wide Gamut RGB* colourspace whitepoint name."""

CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB]
"""*Adobe Wide Gamut RGB* colourspace whitepoint chromaticity coordinates."""

MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_ADOBE_WIDE_GAMUT_RGB, CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB
)
"""*Adobe Wide Gamut RGB* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_ADOBE_WIDE_GAMUT_RGB: NDArrayFloat = np.linalg.inv(
    MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *Adobe Wide Gamut RGB* colourspace matrix."""

RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB: RGB_Colourspace = RGB_Colourspace(
    "Adobe Wide Gamut RGB",
    PRIMARIES_ADOBE_WIDE_GAMUT_RGB,
    CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB,
    WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB,
    MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ,
    MATRIX_XYZ_TO_ADOBE_WIDE_GAMUT_RGB,
    encoding_gamma_function_2_199,
    decoding_gamma_function_2_199,
)
RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB.__doc__ = """
*Adobe Wide Gamut RGB* colourspace.

References
----------
:cite:`Wikipedia2004c`
"""
