# -*- coding: utf-8 -*-
"""
SMPTE C Colourspace
===================

Defines the *SMPTE C* colourspace:

-   :attr:`RGB_COLOURSPACE_SMPTE_C`.

References
----------
-   :cite:`SocietyofMotionPictureandTelevisionEngineers2004a` : Society of
    Motion Picture and Television Engineers. (2004). RP 145:2004: SMPTE C Color
    Monitor Colorimetry. In RP 145:2004: Vol. RP 145:200. The Society of Motion
    Picture and Television Engineers. doi:10.5594/S9781614821649
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
from functools import partial

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, gamma_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_SMPTE_C', 'WHITEPOINT_NAME_SMPTE_C', 'CCS_WHITEPOINT_SMPTE_C',
    'MATRIX_SMPTE_C_TO_XYZ', 'MATRIX_XYZ_TO_SMPTE_C', 'RGB_COLOURSPACE_SMPTE_C'
]

PRIMARIES_SMPTE_C = np.array([
    [0.630, 0.340],
    [0.310, 0.595],
    [0.155, 0.070],
])
"""
*SMPTE C* colourspace primaries.

PRIMARIES_SMPTE_C : ndarray, (3, 2)
"""

WHITEPOINT_NAME_SMPTE_C = 'D65'
"""
*SMPTE C* colourspace whitepoint name.

WHITEPOINT_NAME_SMPTE_C : unicode
"""

CCS_WHITEPOINT_SMPTE_C = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_SMPTE_C])
"""
*SMPTE C* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_SMPTE_C : ndarray
"""

MATRIX_SMPTE_C_TO_XYZ = normalised_primary_matrix(PRIMARIES_SMPTE_C,
                                                  CCS_WHITEPOINT_SMPTE_C)
"""
*SMPTE C* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_SMPTE_C_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_SMPTE_C = np.linalg.inv(MATRIX_SMPTE_C_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *SMPTE C* colourspace matrix.

MATRIX_XYZ_TO_SMPTE_C : array_like, (3, 3)
"""

RGB_COLOURSPACE_SMPTE_C = RGB_Colourspace(
    'SMPTE C',
    PRIMARIES_SMPTE_C,
    CCS_WHITEPOINT_SMPTE_C,
    WHITEPOINT_NAME_SMPTE_C,
    MATRIX_SMPTE_C_TO_XYZ,
    MATRIX_XYZ_TO_SMPTE_C,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
"""
*SMPTE C* colourspace.

References
----------
:cite:`SocietyofMotionPictureandTelevisionEngineers2004a`

RGB_COLOURSPACE_SMPTE_C : RGB_Colourspace
"""
