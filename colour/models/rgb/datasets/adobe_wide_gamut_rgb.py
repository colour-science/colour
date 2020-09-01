# -*- coding: utf-8 -*-
"""
Adobe Wide Gamut RGB Colourspace
================================

Defines the *Adobe Wide Gamut RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB`.

References
----------
-   :cite:`Wikipedia2004c` : Wikipedia. (2004). Wide-gamut RGB color space.
    Retrieved April 13, 2014, from
    http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space
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
    'PRIMARIES_ADOBE_WIDE_GAMUT_RGB', 'WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB',
    'CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB',
    'MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ', 'MATRIX_XYZ_TO_ADOBE_WIDE_GAMUT_RGB',
    'RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB'
]

PRIMARIES_ADOBE_WIDE_GAMUT_RGB = np.array([
    [0.7347, 0.2653],
    [0.1152, 0.8264],
    [0.1566, 0.0177],
])
"""
*Adobe Wide Gamut RGB* colourspace primaries.

PRIMARIES_ADOBE_WIDE_GAMUT_RGB : ndarray, (3, 2)
"""

WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB = 'D50'
"""
*Adobe Wide Gamut RGB* colourspace whitepoint name.

WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB : unicode
"""

CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB = (
    CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
        WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB])
"""
*Adobe Wide Gamut RGB* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB : ndarray
"""

MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ = normalised_primary_matrix(
    PRIMARIES_ADOBE_WIDE_GAMUT_RGB, CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB)
"""
*Adobe Wide Gamut RGB* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_ADOBE_WIDE_GAMUT_RGB = np.linalg.inv(
    MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Adobe Wide Gamut RGB* colourspace matrix.

MATRIX_XYZ_TO_ADOBE_WIDE_GAMUT_RGB : array_like, (3, 3)
"""

RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB = RGB_Colourspace(
    'Adobe Wide Gamut RGB',
    PRIMARIES_ADOBE_WIDE_GAMUT_RGB,
    CCS_WHITEPOINT_ADOBE_WIDE_GAMUT_RGB,
    WHITEPOINT_NAME_ADOBE_WIDE_GAMUT_RGB,
    MATRIX_ADOBE_WIDE_GAMUT_RGB_TO_XYZ,
    MATRIX_XYZ_TO_ADOBE_WIDE_GAMUT_RGB,
    partial(gamma_function, exponent=1 / (563 / 256)),
    partial(gamma_function, exponent=563 / 256),
)
RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB.__doc__ = """
*Adobe Wide Gamut RGB* colourspace.

References
----------
:cite:`Wikipedia2004c`

RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB : RGB_Colourspace
"""
