# -*- coding: utf-8 -*-
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
    'PRIMARIES_COLOR_MATCH_RGB', 'WHITEPOINT_NAME_COLOR_MATCH_RGB',
    'CCS_WHITEPOINT_COLOR_MATCH_RGB', 'MATRIX_COLOR_MATCH_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_COLOR_MATCH_RGB', 'RGB_COLOURSPACE_COLOR_MATCH_RGB'
]

PRIMARIES_COLOR_MATCH_RGB = np.array([
    [0.6300, 0.3400],
    [0.2950, 0.6050],
    [0.1500, 0.0750],
])
"""
*ColorMatch RGB* colourspace primaries.

PRIMARIES_COLOR_MATCH_RGB : ndarray, (3, 2)
"""

WHITEPOINT_NAME_COLOR_MATCH_RGB = 'D50'
"""
*ColorMatch RGB* colourspace whitepoint name.

WHITEPOINT_NAME_COLOR_MATCH_RGB : unicode
"""

CCS_WHITEPOINT_COLOR_MATCH_RGB = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_COLOR_MATCH_RGB])
"""
*ColorMatch RGB* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_COLOR_MATCH_RGB : ndarray
"""

MATRIX_COLOR_MATCH_RGB_TO_XYZ = normalised_primary_matrix(
    PRIMARIES_COLOR_MATCH_RGB, CCS_WHITEPOINT_COLOR_MATCH_RGB)
"""
*ColorMatch RGB* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_COLOR_MATCH_RGB_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_COLOR_MATCH_RGB = np.linalg.inv(MATRIX_COLOR_MATCH_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *ColorMatch RGB* colourspace matrix.

MATRIX_XYZ_TO_COLOR_MATCH_RGB : array_like, (3, 3)
"""

RGB_COLOURSPACE_COLOR_MATCH_RGB = RGB_Colourspace(
    'ColorMatch RGB',
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

RGB_COLOURSPACE_COLOR_MATCH_RGB : RGB_Colourspace
"""
