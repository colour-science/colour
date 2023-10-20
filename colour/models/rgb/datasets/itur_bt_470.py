"""
Recommendation ITU-R BT.470 Colourspaces
========================================

Defines the *Recommendation ITU-R BT.470* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_BT470_525`.
-   :attr:`colour.models.RGB_COLOURSPACE_BT470_625`.

References
----------
-   :cite:`InternationalTelecommunicationUnion1998a` : International
    Telecommunication Union. (1998). Recommendation ITU-R BT.470-6 -
    CONVENTIONAL TELEVISION SYSTEMS (pp. 1-36).
    http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.470-6-199811-S!!PDF-E.pdf
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
    "PRIMARIES_BT470_525",
    "CCS_WHITEPOINT_BT470_525",
    "WHITEPOINT_NAME_BT470_525",
    "MATRIX_BT470_525_TO_XYZ",
    "MATRIX_XYZ_TO_BT470_525",
    "RGB_COLOURSPACE_BT470_525",
    "PRIMARIES_BT470_625",
    "CCS_WHITEPOINT_BT470_625",
    "WHITEPOINT_NAME_BT470_625",
    "MATRIX_BT470_625_TO_XYZ",
    "MATRIX_XYZ_TO_BT470_625",
    "RGB_COLOURSPACE_BT470_625",
]

PRIMARIES_BT470_525: NDArrayFloat = np.array(
    [
        [0.6700, 0.3300],
        [0.2100, 0.7100],
        [0.1400, 0.0800],
    ]
)
"""*Recommendation ITU-R BT.470 - 525* colourspace primaries."""

WHITEPOINT_NAME_BT470_525: str = "C"
"""*Recommendation ITU-R BT.470 - 525* colourspace whitepoint name."""

CCS_WHITEPOINT_BT470_525: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_BT470_525]
"""
*Recommendation ITU-R BT.470 - 525* colourspace whitepoint chromaticity
coordinates.
"""

MATRIX_BT470_525_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_BT470_525, CCS_WHITEPOINT_BT470_525
)
"""
*Recommendation ITU-R BT.470 - 525* colourspace to *CIE XYZ* tristimulus values
matrix.
"""

MATRIX_XYZ_TO_BT470_525: NDArrayFloat = np.linalg.inv(MATRIX_BT470_525_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Recommendation ITU-R BT.470 - 525* colourspace
matrix.
"""

RGB_COLOURSPACE_BT470_525: RGB_Colourspace = RGB_Colourspace(
    "ITU-R BT.470 - 525",
    PRIMARIES_BT470_525,
    CCS_WHITEPOINT_BT470_525,
    WHITEPOINT_NAME_BT470_525,
    MATRIX_BT470_525_TO_XYZ,
    MATRIX_XYZ_TO_BT470_525,
    partial(gamma_function, exponent=1 / 2.8),
    partial(gamma_function, exponent=2.8),
)
RGB_COLOURSPACE_BT470_525.__doc__ = """
*Recommendation ITU-R BT.470 - 525* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion1998a`
"""

PRIMARIES_BT470_625: NDArrayFloat = np.array(
    [
        [0.6400, 0.3300],
        [0.2900, 0.6000],
        [0.1500, 0.0600],
    ]
)
"""*Recommendation ITU-R BT.470 - 625* colourspace primaries."""

WHITEPOINT_NAME_BT470_625: str = "D65"
"""*Recommendation ITU-R BT.470 - 625* colourspace whitepoint name."""

CCS_WHITEPOINT_BT470_625: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_BT470_625]
"""
*Recommendation ITU-R BT.470 - 625* colourspace whitepoint chromaticity
coordinates.
"""

MATRIX_BT470_625_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_BT470_625, CCS_WHITEPOINT_BT470_625
)
"""
*Recommendation ITU-R BT.470 - 625* colourspace to *CIE XYZ* tristimulus values
matrix.
"""

MATRIX_XYZ_TO_BT470_625: NDArrayFloat = np.linalg.inv(MATRIX_BT470_625_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Recommendation ITU-R BT.470 - 625* colourspace
matrix.
"""

RGB_COLOURSPACE_BT470_625: RGB_Colourspace = RGB_Colourspace(
    "ITU-R BT.470 - 625",
    PRIMARIES_BT470_625,
    CCS_WHITEPOINT_BT470_625,
    WHITEPOINT_NAME_BT470_625,
    MATRIX_BT470_625_TO_XYZ,
    MATRIX_XYZ_TO_BT470_625,
    partial(gamma_function, exponent=1 / 2.8),
    partial(gamma_function, exponent=2.8),
)
RGB_COLOURSPACE_BT470_625.__doc__ = """
*Recommendation ITU-R BT.470 - 625* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion1998a`
"""
