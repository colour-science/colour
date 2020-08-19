# -*- coding: utf-8 -*-
"""
P3-D65 Colourspace
==================

Defines the *P3-D65* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_P3_D65`.
"""

from __future__ import division, unicode_literals

import numpy as np
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
    'PRIMARIES_P3_D65', 'WHITEPOINT_NAME_P3_D65', 'CCS_WHITEPOINT_P3_D65',
    'MATRIX_P3_D65_TO_XYZ', 'MATRIX_XYZ_TO_P3_D65', 'RGB_COLOURSPACE_P3_D65'
]

PRIMARIES_P3_D65 = np.array([
    [0.6800, 0.3200],
    [0.2650, 0.6900],
    [0.1500, 0.0600],
])
"""
*P3-D65* colourspace primaries.

PRIMARIES_P3_D65 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_P3_D65 = 'D65'
"""
*P3-D65* colourspace whitepoint name.

WHITEPOINT_NAME_P3_D65 : unicode
"""

CCS_WHITEPOINT_P3_D65 = (CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
                         [WHITEPOINT_NAME_P3_D65])
"""
*P3-D65* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_P3_D65 : ndarray
"""

MATRIX_P3_D65_TO_XYZ = normalised_primary_matrix(PRIMARIES_P3_D65,
                                                 CCS_WHITEPOINT_P3_D65)
"""
*P3-D65* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_P3_D65_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_P3_D65 = np.linalg.inv(MATRIX_P3_D65_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *P3-D65* colourspace matrix.

MATRIX_XYZ_TO_P3_D65 : array_like, (3, 3)
"""

RGB_COLOURSPACE_P3_D65 = RGB_Colourspace(
    'P3-D65',
    PRIMARIES_P3_D65,
    CCS_WHITEPOINT_P3_D65,
    WHITEPOINT_NAME_P3_D65,
    MATRIX_P3_D65_TO_XYZ,
    MATRIX_XYZ_TO_P3_D65,
    partial(gamma_function, exponent=1 / 2.6),
    partial(gamma_function, exponent=2.6),
)
RGB_COLOURSPACE_P3_D65.__doc__ = """
*P3-D65* colourspace.

RGB_COLOURSPACE_P3_D65 : RGB_Colourspace
"""
