# -*- coding: utf-8 -*-
"""
Best RGB Colourspace
====================

Defines the *Best RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_BEST_RGB`.

References
----------
-   :cite:`HutchColord` : HutchColor. (n.d.). BestRGB (4 K).
    http://www.hutchcolor.com/profiles/BestRGB.zip
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
    'PRIMARIES_BEST_RGB', 'WHITEPOINT_NAME_BEST_RGB',
    'CCS_WHITEPOINT_BEST_RGB', 'MATRIX_BEST_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_BEST_RGB', 'RGB_COLOURSPACE_BEST_RGB'
]

PRIMARIES_BEST_RGB = np.array([
    [0.735191637630662, 0.264808362369338],
    [0.215336134453781, 0.774159663865546],
    [0.130122950819672, 0.034836065573770],
])
"""
*Best RGB* colourspace primaries.

PRIMARIES_BEST_RGB : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BEST_RGB = 'D50'
"""
*Best RGB* colourspace whitepoint name.

WHITEPOINT_NAME_BEST_RGB : unicode
"""

CCS_WHITEPOINT_BEST_RGB = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BEST_RGB])
"""
*Best RGB* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_BEST_RGB : ndarray
"""

MATRIX_BEST_RGB_TO_XYZ = normalised_primary_matrix(PRIMARIES_BEST_RGB,
                                                   CCS_WHITEPOINT_BEST_RGB)
"""
*Best RGB* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BEST_RGB_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BEST_RGB = np.linalg.inv(MATRIX_BEST_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Best RGB* colourspace matrix.

MATRIX_XYZ_TO_BEST_RGB : array_like, (3, 3)
"""

RGB_COLOURSPACE_BEST_RGB = RGB_Colourspace(
    'Best RGB',
    PRIMARIES_BEST_RGB,
    CCS_WHITEPOINT_BEST_RGB,
    WHITEPOINT_NAME_BEST_RGB,
    MATRIX_BEST_RGB_TO_XYZ,
    MATRIX_XYZ_TO_BEST_RGB,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_BEST_RGB.__doc__ = """
*Best RGB* colourspace.

References
----------
:cite:`HutchColord`

RGB_COLOURSPACE_BEST_RGB : RGB_Colourspace
"""
