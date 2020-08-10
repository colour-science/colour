# -*- coding: utf-8 -*-
"""
Max RGB Colourspace
===================

Defines the *Max RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_MAX_RGB`.

References
----------
-   :cite:`HutchColorf` : HutchColor. (n.d.). MaxRGB (4 K).
    http://www.hutchcolor.com/profiles/MaxRGB.zip
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
    'PRIMARIES_MAX_RGB', 'WHITEPOINT_NAME_MAX_RGB', 'CCS_WHITEPOINT_MAX_RGB',
    'MATRIX_MAX_RGB_TO_XYZ', 'MATRIX_XYZ_TO_MAX_RGB', 'RGB_COLOURSPACE_MAX_RGB'
]

PRIMARIES_MAX_RGB = np.array([
    [0.73413379, 0.26586621],
    [0.10039113, 0.89960887],
    [0.03621495, 0.00000000],
])
"""
*Max RGB* colourspace primaries.

PRIMARIES_MAX_RGB : ndarray, (3, 2)
"""

WHITEPOINT_NAME_MAX_RGB = 'D50'
"""
*Max RGB* colourspace whitepoint name.

WHITEPOINT_NAME_MAX_RGB : unicode
"""

CCS_WHITEPOINT_MAX_RGB = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_MAX_RGB])
"""
*Max RGB* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_MAX_RGB : ndarray
"""

MATRIX_MAX_RGB_TO_XYZ = normalised_primary_matrix(PRIMARIES_MAX_RGB,
                                                  CCS_WHITEPOINT_MAX_RGB)
"""
*Max RGB* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_MAX_RGB_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_MAX_RGB = np.linalg.inv(MATRIX_MAX_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Max RGB* colourspace matrix.

MATRIX_XYZ_TO_MAX_RGB : array_like, (3, 3)
"""

RGB_COLOURSPACE_MAX_RGB = RGB_Colourspace(
    'Max RGB',
    PRIMARIES_MAX_RGB,
    CCS_WHITEPOINT_MAX_RGB,
    WHITEPOINT_NAME_MAX_RGB,
    MATRIX_MAX_RGB_TO_XYZ,
    MATRIX_XYZ_TO_MAX_RGB,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_MAX_RGB.__doc__ = """
*Max RGB* colourspace.

References
----------
:cite:`HutchColorf`

RGB_COLOURSPACE_MAX_RGB : RGB_Colourspace
"""
