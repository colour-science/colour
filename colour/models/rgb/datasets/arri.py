"""
ARRI Colourspaces
=================

Defines the *ARRI* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3`.

References
----------
-   :cite:`ARRI2012a` : ARRI. (2012). ALEXA - Log C Curve - Usage in VFX.
    https://drive.google.com/open?id=1t73fAG_QpV7hJxoQPYZDWvOojYkYDgvn
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    log_encoding_ARRILogC3,
    log_decoding_ARRILogC3,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_ARRI_WIDE_GAMUT_3",
    "WHITEPOINT_NAME_ARRI_WIDE_GAMUT_3",
    "CCS_WHITEPOINT_ARRI_WIDE_GAMUT_3",
    "MATRIX_ARRI_WIDE_GAMUT_3_TO_XYZ",
    "MATRIX_XYZ_TO_ARRI_WIDE_GAMUT_3",
    "RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3",
]

PRIMARIES_ARRI_WIDE_GAMUT_3: NDArray = np.array(
    [
        [0.6840, 0.3130],
        [0.2210, 0.8480],
        [0.0861, -0.1020],
    ]
)
"""*ARRI Wide Gamut 3* colourspace primaries."""

WHITEPOINT_NAME_ARRI_WIDE_GAMUT_3: str = "D65"
"""*ARRI Wide Gamut 3* colourspace whitepoint name."""

CCS_WHITEPOINT_ARRI_WIDE_GAMUT_3: NDArray = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_ARRI_WIDE_GAMUT_3]
"""*ARRI Wide Gamut 3* colourspace whitepoint chromaticity coordinates."""

MATRIX_ARRI_WIDE_GAMUT_3_TO_XYZ: NDArray = np.array(
    [
        [0.638008, 0.214704, 0.097744],
        [0.291954, 0.823841, -0.115795],
        [0.002798, -0.067034, 1.153294],
    ]
)
"""*ARRI Wide Gamut 3* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_ARRI_WIDE_GAMUT_3: NDArray = np.array(
    [
        [1.789066, -0.482534, -0.200076],
        [-0.639849, 1.396400, 0.194432],
        [-0.041532, 0.082335, 0.878868],
    ]
)
"""*CIE XYZ* tristimulus values to *ARRI Wide Gamut 3* colourspace matrix."""

RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3: RGB_Colourspace = RGB_Colourspace(
    "ARRI Wide Gamut 3",
    PRIMARIES_ARRI_WIDE_GAMUT_3,
    CCS_WHITEPOINT_ARRI_WIDE_GAMUT_3,
    WHITEPOINT_NAME_ARRI_WIDE_GAMUT_3,
    MATRIX_ARRI_WIDE_GAMUT_3_TO_XYZ,
    MATRIX_XYZ_TO_ARRI_WIDE_GAMUT_3,
    log_encoding_ARRILogC3,
    log_decoding_ARRILogC3,
)
RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3.__doc__ = """
*ARRI Wide Gamut 3* colourspace.

References
----------
:cite:`ARRI2012a`
"""
