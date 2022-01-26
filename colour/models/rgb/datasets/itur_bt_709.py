# -*- coding: utf-8 -*-
"""
ITU-R BT.709 Colourspace
========================

Defines the *ITU-R BT.709* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_BT709`.

References
----------
-   :cite:`InternationalTelecommunicationUnion2015i` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.709-6 - Parameter
    values for the HDTV standards for production and international programme
    exchange BT Series Broadcasting service (pp. 1-32).
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    oetf_BT709,
    oetf_inverse_BT709,
    normalised_primary_matrix,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_BT709',
    'CCS_WHITEPOINT_BT709',
    'WHITEPOINT_NAME_BT709',
    'MATRIX_BT709_TO_XYZ',
    'MATRIX_XYZ_TO_BT709',
    'RGB_COLOURSPACE_BT709',
]

PRIMARIES_BT709: NDArray = np.array([
    [0.6400, 0.3300],
    [0.3000, 0.6000],
    [0.1500, 0.0600],
])
"""
*ITU-R BT.709* colourspace primaries.
"""

WHITEPOINT_NAME_BT709: str = 'D65'
"""
*ITU-R BT.709* colourspace whitepoint name.
"""

CCS_WHITEPOINT_BT709: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BT709])
"""
*ITU-R BT.709* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_BT709_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_BT709, CCS_WHITEPOINT_BT709)
"""
*ITU-R BT.709* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_BT709: NDArray = np.linalg.inv(MATRIX_BT709_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *ITU-R BT.709* colourspace matrix.
"""

RGB_COLOURSPACE_BT709: RGB_Colourspace = RGB_Colourspace(
    'ITU-R BT.709',
    PRIMARIES_BT709,
    CCS_WHITEPOINT_BT709,
    WHITEPOINT_NAME_BT709,
    MATRIX_BT709_TO_XYZ,
    MATRIX_XYZ_TO_BT709,
    oetf_BT709,
    oetf_inverse_BT709,
)
RGB_COLOURSPACE_BT709.__doc__ = """
*ITU-R BT.709* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion2015i`
"""
