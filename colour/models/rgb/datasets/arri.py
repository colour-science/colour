"""
ARRI Colourspaces
=================

Defines the *ARRI* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3`.
-   :attr:`colour.models.RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4`.

References
----------
-   :cite:`ARRI2012a` : ARRI. (2012). ALEXA - Log C Curve - Usage in VFX.
    https://drive.google.com/open?id=1t73fAG_QpV7hJxoQPYZDWvOojYkYDgvn
-   :cite:`Cooper2022` : Cooper, S., & Brendel, H. (2022). ARRI LogC4
    Logarithmic Color Space SPECIFICATION. Retrieved October 24, 2022, from
    https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/\
2022-05-arri-logc4-specification-data.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArrayFloat
from colour.models.rgb import (
    RGB_Colourspace,
    log_decoding_ARRILogC3,
    log_decoding_ARRILogC4,
    log_encoding_ARRILogC3,
    log_encoding_ARRILogC4,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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
    "PRIMARIES_ARRI_WIDE_GAMUT_4",
    "WHITEPOINT_NAME_ARRI_WIDE_GAMUT_4",
    "CCS_WHITEPOINT_ARRI_WIDE_GAMUT_4",
    "MATRIX_ARRI_WIDE_GAMUT_4_TO_XYZ",
    "MATRIX_XYZ_TO_ARRI_WIDE_GAMUT_4",
    "RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4",
]

PRIMARIES_ARRI_WIDE_GAMUT_3: NDArrayFloat = np.array(
    [
        [0.6840, 0.3130],
        [0.2210, 0.8480],
        [0.0861, -0.1020],
    ]
)
"""*ARRI Wide Gamut 3* colourspace primaries."""

WHITEPOINT_NAME_ARRI_WIDE_GAMUT_3: str = "D65"
"""*ARRI Wide Gamut 3* colourspace whitepoint name."""

CCS_WHITEPOINT_ARRI_WIDE_GAMUT_3: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_ARRI_WIDE_GAMUT_3]
"""*ARRI Wide Gamut 3* colourspace whitepoint chromaticity coordinates."""

MATRIX_ARRI_WIDE_GAMUT_3_TO_XYZ: NDArrayFloat = np.array(
    [
        [0.638008, 0.214704, 0.097744],
        [0.291954, 0.823841, -0.115795],
        [0.002798, -0.067034, 1.153294],
    ]
)
"""*ARRI Wide Gamut 3* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_ARRI_WIDE_GAMUT_3: NDArrayFloat = np.array(
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

PRIMARIES_ARRI_WIDE_GAMUT_4: NDArrayFloat = np.array(
    [
        [0.7347, 0.2653],
        [0.1424, 0.8576],
        [0.0991, -0.0308],
    ]
)
"""*ARRI Wide Gamut 4* colourspace primaries."""

WHITEPOINT_NAME_ARRI_WIDE_GAMUT_4: str = "D65"
"""*ARRI Wide Gamut 4* colourspace whitepoint name."""

CCS_WHITEPOINT_ARRI_WIDE_GAMUT_4: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_ARRI_WIDE_GAMUT_4]
"""*ARRI Wide Gamut 4* colourspace whitepoint chromaticity coordinates."""

MATRIX_ARRI_WIDE_GAMUT_4_TO_XYZ: NDArrayFloat = np.array(
    [
        [0.7048583204, 0.1297602952, 0.1158373115],
        [0.2545241764, 0.7814777327, -0.0360019091],
        [0.0000000000, 0.0000000000, 1.0890577508],
    ]
)
"""*ARRI Wide Gamut 4* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_ARRI_WIDE_GAMUT_4: NDArrayFloat = np.linalg.inv(
    MATRIX_ARRI_WIDE_GAMUT_4_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *ARRI Wide Gamut 4* colourspace matrix."""

RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4: RGB_Colourspace = RGB_Colourspace(
    "ARRI Wide Gamut 4",
    PRIMARIES_ARRI_WIDE_GAMUT_4,
    CCS_WHITEPOINT_ARRI_WIDE_GAMUT_4,
    WHITEPOINT_NAME_ARRI_WIDE_GAMUT_4,
    MATRIX_ARRI_WIDE_GAMUT_4_TO_XYZ,
    MATRIX_XYZ_TO_ARRI_WIDE_GAMUT_4,
    log_encoding_ARRILogC4,
    log_decoding_ARRILogC4,
)
RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4.__doc__ = """
*ARRI Wide Gamut 4* colourspace.

References
----------
:cite:`Cooper2022`
"""
