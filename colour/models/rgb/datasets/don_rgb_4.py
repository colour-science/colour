# -*- coding: utf-8 -*-
"""
Don RGB 4 Colourspace
=====================

Defines the *Don RGB 4* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_DON_RGB_4`.

References
----------
-   :cite:`HutchColorg` : HutchColor. (n.d.). DonRGB4 (4 K).
    http://www.hutchcolor.com/profiles/DonRGB4.zip
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
    'PRIMARIES_DON_RGB_4', 'WHITEPOINT_NAME_DON_RGB_4',
    'CCS_WHITEPOINT_DON_RGB_4', 'MATRIX_DON_RGB_4_TO_XYZ',
    'MATRIX_XYZ_TO_DON_RGB_4', 'RGB_COLOURSPACE_DON_RGB_4'
]

PRIMARIES_DON_RGB_4 = np.array([
    [0.696120689655172, 0.299568965517241],
    [0.214682981090100, 0.765294771968854],
    [0.129937629937630, 0.035343035343035],
])
"""
*Don RGB 4* colourspace primaries.

PRIMARIES_DON_RGB_4 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_DON_RGB_4 = 'D50'
"""
*Don RGB 4* colourspace whitepoint name.

WHITEPOINT_NAME_DON_RGB_4 : unicode
"""

CCS_WHITEPOINT_DON_RGB_4 = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_DON_RGB_4])
"""
*Don RGB 4* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_DON_RGB_4 : ndarray
"""

MATRIX_DON_RGB_4_TO_XYZ = normalised_primary_matrix(PRIMARIES_DON_RGB_4,
                                                    CCS_WHITEPOINT_DON_RGB_4)
"""
*Don RGB 4* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_DON_RGB_4_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_DON_RGB_4 = np.linalg.inv(MATRIX_DON_RGB_4_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Don RGB 4* colourspace matrix.

MATRIX_XYZ_TO_DON_RGB_4 : array_like, (3, 3)
"""

RGB_COLOURSPACE_DON_RGB_4 = RGB_Colourspace(
    'Don RGB 4',
    PRIMARIES_DON_RGB_4,
    CCS_WHITEPOINT_DON_RGB_4,
    WHITEPOINT_NAME_DON_RGB_4,
    MATRIX_DON_RGB_4_TO_XYZ,
    MATRIX_XYZ_TO_DON_RGB_4,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_DON_RGB_4.__doc__ = """
*Don RGB 4* colourspace.

References
----------
:cite:`HutchColorg`

RGB_COLOURSPACE_DON_RGB_4 : RGB_Colourspace
"""
