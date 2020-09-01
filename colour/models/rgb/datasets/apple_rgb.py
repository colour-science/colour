# -*- coding: utf-8 -*-
"""
Apple RGB Colourspace
=====================

Defines the *Apple RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_APPLE_RGB`.

References
----------
-   :cite:`Susstrunk1999a` : Susstrunk, S., Buckley, R., & Swen, S. (1999).
    Standard RGB Color Spaces.
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
    'PRIMARIES_APPLE_RGB', 'WHITEPOINT_NAME_APPLE_RGB',
    'CCS_WHITEPOINT_APPLE_RGB', 'MATRIX_APPLE_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_APPLE_RGB', 'RGB_COLOURSPACE_APPLE_RGB'
]

PRIMARIES_APPLE_RGB = np.array([
    [0.6250, 0.3400],
    [0.2800, 0.5950],
    [0.1550, 0.0700],
])
"""
*Apple RGB* colourspace primaries.

PRIMARIES_APPLE_RGB : ndarray, (3, 2)
"""

WHITEPOINT_NAME_APPLE_RGB = 'D65'
"""
*Apple RGB* colourspace whitepoint name.

WHITEPOINT_NAME_APPLE_RGB : unicode
"""

CCS_WHITEPOINT_APPLE_RGB = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_APPLE_RGB])
"""
*Apple RGB* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_APPLE_RGB : ndarray
"""

MATRIX_APPLE_RGB_TO_XYZ = normalised_primary_matrix(PRIMARIES_APPLE_RGB,
                                                    CCS_WHITEPOINT_APPLE_RGB)
"""
*Apple RGB* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_APPLE_RGB_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_APPLE_RGB = np.linalg.inv(MATRIX_APPLE_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Apple RGB* colourspace matrix.

MATRIX_XYZ_TO_APPLE_RGB : array_like, (3, 3)
"""

RGB_COLOURSPACE_APPLE_RGB = RGB_Colourspace(
    'Apple RGB',
    PRIMARIES_APPLE_RGB,
    CCS_WHITEPOINT_APPLE_RGB,
    WHITEPOINT_NAME_APPLE_RGB,
    MATRIX_APPLE_RGB_TO_XYZ,
    MATRIX_XYZ_TO_APPLE_RGB,
    partial(gamma_function, exponent=1 / 1.8),
    partial(gamma_function, exponent=1.8),
)
RGB_COLOURSPACE_APPLE_RGB.__doc__ = """
*Apple RGB* colourspace.

References
----------
:cite:`Susstrunk1999a`

RGB_COLOURSPACE_APPLE_RGB : RGB_Colourspace
"""
