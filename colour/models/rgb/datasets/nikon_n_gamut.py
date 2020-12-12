# -*- coding: utf-8 -*-
"""
Nikon N-Gamut Colourspace
=========================

Defines the *Nikon N-Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_N_GAMUT`.

References
----------
-   :cite:`Nikon2018` : Nikon. (2018). N-Log Specification Document - Version
    1.0.0 (pp. 1–5). Retrieved September 9, 2019, from
    http://download.nikonimglib.com/archive3/hDCmK00m9JDI03RPruD74xpoU905/\
N-Log_Specification_(En)01.pdf
"""

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_encoding_NLog,
                               normalised_primary_matrix, log_decoding_NLog)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_N_GAMUT', 'WHITEPOINT_NAME_N_GAMUT', 'WHITEPOINT_N_GAMUT',
    'MATRIX_N_GAMUT_TO_XYZ', 'MATRIX_XYZ_TO_N_GAMUT', 'RGB_COLOURSPACE_N_GAMUT'
]

PRIMARIES_N_GAMUT = np.array([
    [0.70800, 0.29200],
    [0.17000, 0.79700],
    [0.13100, 0.04600],
])
"""
*Nikon N-Gamut* colourspace primaries.

PRIMARIES_N_GAMUT : ndarray, (3, 2)
"""

WHITEPOINT_NAME_N_GAMUT = 'D65'
"""
*Nikon N-Gamut* colourspace whitepoint name.

WHITEPOINT_N_GAMUT : unicode
"""

WHITEPOINT_N_GAMUT = (CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    WHITEPOINT_NAME_N_GAMUT])
"""
*Nikon N-Gamut* colourspace whitepoint.

WHITEPOINT_N_GAMUT : ndarray
"""

MATRIX_N_GAMUT_TO_XYZ = normalised_primary_matrix(PRIMARIES_N_GAMUT,
                                                  WHITEPOINT_N_GAMUT)
"""
*Nikon N-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_N_GAMUT_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_N_GAMUT = np.linalg.inv(MATRIX_N_GAMUT_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Nikon N-Gamut* colourspace matrix.

MATRIX_XYZ_TO_N_GAMUT : array_like, (3, 3)
"""

RGB_COLOURSPACE_N_GAMUT = RGB_Colourspace(
    'N-Gamut',
    PRIMARIES_N_GAMUT,
    WHITEPOINT_N_GAMUT,
    WHITEPOINT_NAME_N_GAMUT,
    MATRIX_N_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_N_GAMUT,
    log_encoding_NLog,
    log_decoding_NLog,
)
RGB_COLOURSPACE_N_GAMUT.__doc__ = """
*Nikon N-Gamut* colourspace.

References
----------
:cite:`Nikon2018`

RGB_COLOURSPACE_N_GAMUT : RGB_Colourspace
"""
