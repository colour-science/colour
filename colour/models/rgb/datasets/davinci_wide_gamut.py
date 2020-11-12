# -*- coding: utf-8 -*-
"""
DaVinci Wide Gamut Colourspace
==============================

Defines the *DaVinci Wide Gamut* *RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT`.

References
----------
-   :cite:`BlackmagicDesign2020` : Blackmagic Design. (2020).
    DaVinci Wide Gamut - DaVinci Resolve Studio 17 Public Beta 1.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, linear_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_DAVINCI_WIDE_GAMUT', 'WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT',
    'CCS_WHITEPOINT_DAVINCI_WIDE_GAMUT', 'MATRIX_DAVINCI_WIDE_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_DAVINCI_WIDE_GAMUT', 'RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT'
]

PRIMARIES_DAVINCI_WIDE_GAMUT = np.array([
    [0.8000, 0.3130],
    [0.1682, 0.9877],
    [0.0790, -0.1155],
])
"""
*DaVinci Wide Gamut* colourspace primaries.

PRIMARIES_DAVINCI_WIDE_GAMUT : ndarray, (3, 2)
"""

WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT = 'D65'
"""
*DaVinci Wide Gamut* colourspace whitepoint name.

WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT : unicode
"""

CCS_WHITEPOINT_DAVINCI_WIDE_GAMUT = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT])
"""
*DaVinci Wide Gamut* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_DAVINCI_WIDE_GAMUT : ndarray
"""

MATRIX_DAVINCI_WIDE_GAMUT_TO_XYZ = normalised_primary_matrix(
    PRIMARIES_DAVINCI_WIDE_GAMUT, CCS_WHITEPOINT_DAVINCI_WIDE_GAMUT)
"""
*DaVinci Wide Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_DAVINCI_WIDE_GAMUT_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_DAVINCI_WIDE_GAMUT = np.linalg.inv(
    MATRIX_DAVINCI_WIDE_GAMUT_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *DaVinci Wide Gamut* colourspace matrix.

MATRIX_XYZ_TO_DAVINCI_WIDE_GAMUT : array_like, (3, 3)
"""

RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT = RGB_Colourspace(
    'DaVinci Wide Gamut',
    PRIMARIES_DAVINCI_WIDE_GAMUT,
    CCS_WHITEPOINT_DAVINCI_WIDE_GAMUT,
    WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT,
    MATRIX_DAVINCI_WIDE_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_DAVINCI_WIDE_GAMUT,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT.__doc__ = """
*DaVinci Wide Gamut* colourspace.

References
----------
:cite:`BlackmagicDesign2020`

RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT : RGB_Colourspace
"""
