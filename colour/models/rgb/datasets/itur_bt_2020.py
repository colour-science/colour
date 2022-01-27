# -*- coding: utf-8 -*-
"""
ITU-R BT.2020 Colourspace
=========================

Defines the *ITU-R BT.2020* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_BT2020`.

References
----------
-   :cite:`InternationalTelecommunicationUnion2015h` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.2020 - Parameter
    values for ultra-high definition television systems for production and
    international programme exchange (pp. 1-8).
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2020-2-201510-I!!PDF-E.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    normalised_primary_matrix,
    eotf_inverse_BT2020,
    eotf_BT2020,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_BT2020',
    'WHITEPOINT_NAME_BT2020',
    'CCS_WHITEPOINT_BT2020',
    'MATRIX_BT2020_TO_XYZ',
    'MATRIX_XYZ_TO_BT2020',
    'RGB_COLOURSPACE_BT2020',
]

PRIMARIES_BT2020: NDArray = np.array([
    [0.7080, 0.2920],
    [0.1700, 0.7970],
    [0.1310, 0.0460],
])
"""
*ITU-R BT.2020* colourspace primaries.
"""

WHITEPOINT_NAME_BT2020: str = 'D65'
"""
*ITU-R BT.2020* colourspace whitepoint name.
"""

CCS_WHITEPOINT_BT2020: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BT2020])
"""
*ITU-R BT.2020* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_BT2020_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_BT2020, CCS_WHITEPOINT_BT2020)
"""
*ITU-R BT.2020* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_BT2020: NDArray = np.linalg.inv(MATRIX_BT2020_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *ITU-R BT.2020* colourspace matrix.
"""

RGB_COLOURSPACE_BT2020: RGB_Colourspace = RGB_Colourspace(
    'ITU-R BT.2020',
    PRIMARIES_BT2020,
    CCS_WHITEPOINT_BT2020,
    WHITEPOINT_NAME_BT2020,
    MATRIX_BT2020_TO_XYZ,
    MATRIX_XYZ_TO_BT2020,
    eotf_inverse_BT2020,
    eotf_BT2020,
)
RGB_COLOURSPACE_BT2020.__doc__ = """
*ITU-R BT.2020* colourspace.

The wavelength of the *ITU-R BT.2020* primary colours are:

-   630nm for the red primary colour
-   532nm for the green primary colour
-   467nm for the blue primary colour.

References
----------
:cite:`InternationalTelecommunicationUnion2015h`
"""
