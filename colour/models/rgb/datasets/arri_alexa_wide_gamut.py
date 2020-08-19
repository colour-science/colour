# -*- coding: utf-8 -*-
"""
ARRI ALEXA Wide Gamut Colourspace
=================================

Defines the *ARRI ALEXA Wide Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_ALEXA_WIDE_GAMUT`.

References
----------
-   :cite:`ARRI2012a` : ARRI. (2012). ALEXA - Log C Curve - Usage in VFX.
    https://drive.google.com/open?id=1t73fAG_QpV7hJxoQPYZDWvOojYkYDgvn
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_encoding_ALEXALogC,
                               log_decoding_ALEXALogC)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_ALEXA_WIDE_GAMUT', 'WHITEPOINT_NAME_ALEXA_WIDE_GAMUT',
    'CCS_WHITEPOINT_ALEXA_WIDE_GAMUT', 'MATRIX_ALEXA_WIDE_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_ALEXA_WIDE_GAMUT', 'RGB_COLOURSPACE_ALEXA_WIDE_GAMUT'
]

PRIMARIES_ALEXA_WIDE_GAMUT = np.array([
    [0.6840, 0.3130],
    [0.2210, 0.8480],
    [0.0861, -0.1020],
])
"""
*ARRI ALEXA Wide Gamut* colourspace primaries.

PRIMARIES_ALEXA_WIDE_GAMUT : ndarray, (3, 2)
"""

WHITEPOINT_NAME_ALEXA_WIDE_GAMUT = 'D65'
"""
*ARRI ALEXA Wide Gamut* colourspace whitepoint name.

CCS_WHITEPOINT_ALEXA_WIDE_GAMUT : unicode
"""

CCS_WHITEPOINT_ALEXA_WIDE_GAMUT = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_ALEXA_WIDE_GAMUT])
"""
*ARRI ALEXA Wide Gamut* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_ALEXA_WIDE_GAMUT : ndarray
"""

MATRIX_ALEXA_WIDE_GAMUT_TO_XYZ = np.array([
    [0.638008, 0.214704, 0.097744],
    [0.291954, 0.823841, -0.115795],
    [0.002798, -0.067034, 1.153294],
])
"""
*ARRI ALEXA Wide Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_ALEXA_WIDE_GAMUT_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_ALEXA_WIDE_GAMUT = np.array([
    [1.789066, -0.482534, -0.200076],
    [-0.639849, 1.396400, 0.194432],
    [-0.041532, 0.082335, 0.878868],
])
"""
*CIE XYZ* tristimulus values to *ARRI ALEXA Wide Gamut* colourspace matrix.

MATRIX_XYZ_TO_ALEXA_WIDE_GAMUT : array_like, (3, 3)
"""

RGB_COLOURSPACE_ALEXA_WIDE_GAMUT = RGB_Colourspace(
    'ALEXA Wide Gamut',
    PRIMARIES_ALEXA_WIDE_GAMUT,
    CCS_WHITEPOINT_ALEXA_WIDE_GAMUT,
    WHITEPOINT_NAME_ALEXA_WIDE_GAMUT,
    MATRIX_ALEXA_WIDE_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_ALEXA_WIDE_GAMUT,
    log_encoding_ALEXALogC,
    log_decoding_ALEXALogC,
)
RGB_COLOURSPACE_ALEXA_WIDE_GAMUT.__doc__ = """
*ARRI ALEXA Wide Gamut* colourspace.

References
----------
:cite:`ARRI2012a`

RGB_COLOURSPACE_ALEXA_WIDE_GAMUT : RGB_Colourspace
"""
