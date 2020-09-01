# -*- coding: utf-8 -*-
"""
Display P3 Colourspace
======================

Defines the *Display P3* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_DISPLAY_P3`.

References
----------
-   :cite:`AppleInc.2019` : Apple. (2019). Apple Inc. (2019). displayP3.
    Retrieved December 18, 2019, from https://developer.apple.com/\
documentation/coregraphics/cgcolorspace/1408916-displayp3
"""

from __future__ import division, unicode_literals

import colour.ndarray as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, eotf_inverse_sRGB, eotf_sRGB,
                               normalised_primary_matrix)
from colour.models.rgb.datasets import RGB_COLOURSPACE_DCI_P3

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_DISPLAY_P3', 'WHITEPOINT_NAME_DISPLAY_P3',
    'CCS_WHITEPOINT_DISPLAY_P3', 'MATRIX_DISPLAY_P3_TO_XYZ',
    'MATRIX_XYZ_TO_DISPLAY_P3', 'RGB_COLOURSPACE_DISPLAY_P3'
]

PRIMARIES_DISPLAY_P3 = RGB_COLOURSPACE_DCI_P3.primaries
"""
*Display P3* colourspace primaries.

PRIMARIES_DISPLAY_P3 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_DISPLAY_P3 = 'D65'
"""
*Display P3* colourspace whitepoint name.

CCS_WHITEPOINT_DISPLAY_P3 : unicode
"""

CCS_WHITEPOINT_DISPLAY_P3 = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_DISPLAY_P3])
"""
*Display P3* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_DISPLAY_P3 : ndarray
"""

MATRIX_DISPLAY_P3_TO_XYZ = (normalised_primary_matrix(
    PRIMARIES_DISPLAY_P3, CCS_WHITEPOINT_DISPLAY_P3))
"""
*Display P3* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_DISPLAY_P3_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_DISPLAY_P3 = np.linalg.inv(MATRIX_DISPLAY_P3_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Display P3* colourspace matrix.

MATRIX_XYZ_TO_DISPLAY_P3 : array_like, (3, 3)
"""

RGB_COLOURSPACE_DISPLAY_P3 = RGB_Colourspace(
    'Display P3',
    PRIMARIES_DISPLAY_P3,
    CCS_WHITEPOINT_DISPLAY_P3,
    WHITEPOINT_NAME_DISPLAY_P3,
    MATRIX_DISPLAY_P3_TO_XYZ,
    MATRIX_XYZ_TO_DISPLAY_P3,
    eotf_inverse_sRGB,
    eotf_sRGB,
)
RGB_COLOURSPACE_DISPLAY_P3.__doc__ = """
*Display P3* colourspace.

References
----------
:cite:`AppleInc.2019`

RGB_COLOURSPACE_DISPLAY_P3 : RGB_Colourspace
"""
