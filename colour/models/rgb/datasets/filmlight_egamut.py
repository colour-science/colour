# -*- coding: utf-8 -*-
"""
FilmLight E-Gamut Colourspace
=============================

Defines the *FilmLight E-Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_FILMLIGHT_E_GAMUT`.

References
----------
-   :cite:`Siragusano2018a` : Siragusano, D. (2018). Private Discussion with
    Shaw, Nick.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_encoding_FilmLightTLog,
                               log_decoding_FilmLightTLog,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_FILMLIGHT_E_GAMUT', 'WHITEPOINT_NAME_FILMLIGHT_E_GAMUT',
    'CCS_WHITEPOINT_FILMLIGHT_E_GAMUT', 'MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT', 'RGB_COLOURSPACE_FILMLIGHT_E_GAMUT'
]

PRIMARIES_FILMLIGHT_E_GAMUT = np.array([
    [0.8000, 0.3177],
    [0.1800, 0.9000],
    [0.0650, -0.0805],
])
"""
*FilmLight E-Gamut* colourspace primaries.

PRIMARIES_FILMLIGHT_E_GAMUT : ndarray, (3, 2)
"""

WHITEPOINT_NAME_FILMLIGHT_E_GAMUT = 'D65'
"""
*FilmLight E-Gamut* colourspace whitepoint name.

CCS_WHITEPOINT_FILMLIGHT_E_GAMUT : unicode
"""

CCS_WHITEPOINT_FILMLIGHT_E_GAMUT = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_FILMLIGHT_E_GAMUT])
"""
*FilmLight E-Gamut* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_FILMLIGHT_E_GAMUT : ndarray
"""

MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ = (normalised_primary_matrix(
    PRIMARIES_FILMLIGHT_E_GAMUT, CCS_WHITEPOINT_FILMLIGHT_E_GAMUT))
"""
*FilmLight E-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT = (
    np.linalg.inv(MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ))
"""
*CIE XYZ* tristimulus values to *FilmLight E-Gamut* colourspace matrix.

MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT : array_like, (3, 3)
"""

RGB_COLOURSPACE_FILMLIGHT_E_GAMUT = RGB_Colourspace(
    'FilmLight E-Gamut',
    PRIMARIES_FILMLIGHT_E_GAMUT,
    CCS_WHITEPOINT_FILMLIGHT_E_GAMUT,
    WHITEPOINT_NAME_FILMLIGHT_E_GAMUT,
    MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT,
    log_encoding_FilmLightTLog,
    log_decoding_FilmLightTLog,
)
RGB_COLOURSPACE_FILMLIGHT_E_GAMUT.__doc__ = """
*FilmLight E-Gamut* colourspace.

    References
    ----------
    :cite:`Siragusano2018a`

RGB_COLOURSPACE_FILMLIGHT_E_GAMUT : RGB_Colourspace
"""
