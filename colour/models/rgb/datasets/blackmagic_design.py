# -*- coding: utf-8 -*-
"""
Blackmagic Design Colourspaces
==============================

Defines the *Blackmagic Design* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_BMD_FILM_V1`.
-   :attr:`colour.models.RGB_COLOURSPACE_BMD_4K_FILM_V1`.
-   :attr:`colour.models.RGB_COLOURSPACE_BMD_4K_FILM_V3`.
-   :attr:`colour.models.RGB_COLOURSPACE_BMD_46K_FILM_V1`.
-   :attr:`colour.models.RGB_COLOURSPACE_BMD_46K_FILM_V3`.
-   :attr:`colour.models.RGB_COLOURSPACE_BMD_WIDE_GAMUT_V4`.

References
----------
-   :cite:`Blackmagic2020a` : Blackmagic Design. (2020). DaVinci Resolve
    CIE Chromaticity Plot.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    log_encoding_BMDFilm,
    log_decoding_BMDFilm,
    log_encoding_BMD4KFilm,
    log_decoding_BMD4KFilm,
    log_encoding_BMD46KFilm,
    log_decoding_BMD46KFilm,
    log_encoding_BMDPocket4KFilmV4,
    log_decoding_BMDPocket4KFilmV4,
    log_encoding_BMDPocket6KFilmV4,
    log_decoding_BMDPocket6KFilmV4,
    normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_BMD_FILM_V1', 'WHITEPOINT_NAME_BMD_FILM_V1',
    'WHITEPOINT_BMD_FILM_V1', 'MATRIX_BMD_FILM_V1_TO_XYZ',
    'MATRIX_XYZ_TO_BMD_FILM_V1', 'RGB_COLOURSPACE_BMD_FILM_V1',
    'PRIMARIES_BMD_4K_FILM_V1', 'WHITEPOINT_NAME_BMD_4K_FILM_V1',
    'WHITEPOINT_BMD_4K_FILM_V1', 'MATRIX_BMD_4K_FILM_V1_TO_XYZ',
    'MATRIX_XYZ_TO_BMD_4K_FILM_V1', 'RGB_COLOURSPACE_BMD_4K_FILM_V1',
    'PRIMARIES_BMD_4K_FILM_V3', 'WHITEPOINT_NAME_BMD_4K_FILM_V3',
    'WHITEPOINT_BMD_4K_FILM_V3', 'MATRIX_BMD_4K_FILM_V3_TO_XYZ',
    'MATRIX_XYZ_TO_BMD_4K_FILM_V3', 'RGB_COLOURSPACE_BMD_4K_FILM_V3',
    'PRIMARIES_BMD_46K_FILM_V1', 'WHITEPOINT_NAME_BMD_46K_FILM_V1',
    'WHITEPOINT_BMD_46K_FILM_V1', 'MATRIX_BMD_46K_FILM_V1_TO_XYZ',
    'MATRIX_XYZ_TO_BMD_46K_FILM_V1', 'RGB_COLOURSPACE_BMD_46K_FILM_V1',
    'PRIMARIES_BMD_46K_FILM_V3', 'WHITEPOINT_NAME_BMD_46K_FILM_V3',
    'WHITEPOINT_BMD_46K_FILM_V3', 'MATRIX_BMD_46K_FILM_V3_TO_XYZ',
    'MATRIX_XYZ_TO_BMD_46K_FILM_V3', 'RGB_COLOURSPACE_BMD_46K_FILM_V3',
    'PRIMARIES_BMD_WIDE_GAMUT_V4', 'WHITEPOINT_NAME_BMD_WIDE_GAMUT_V4',
    'WHITEPOINT_BMD_WIDE_GAMUT_V4', 'MATRIX_BMD_WIDE_GAMUT_V4_TO_XYZ',
    'MATRIX_XYZ_TO_BMD_WIDE_GAMUT_V4', 'RGB_COLOURSPACE_BMD_WIDE_GAMUT_V4'
]

PRIMARIES_BMD_FILM_V1 = np.array([
    [0.9173, 0.2502],
    [0.2833, 1.7072],
    [0.0856, -0.0708],
])
"""
*BMD Film V1* colourspace primaries.

PRIMARIES_BMD_FILM_V1 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BMD_FILM_V1 = 'BMD White'
"""
*BMD Film V1* colourspace whitepoint name.

WHITEPOINT_BMD_FILM_V1 : unicode
"""

WHITEPOINT_BMD_FILM_V1 = np.array([0.3135, 0.3305])
"""
*BMD Film V1* colourspace whitepoint.

WHITEPOINT_BMD_FILM_V1 : ndarray
"""

MATRIX_BMD_FILM_V1_TO_XYZ = (normalised_primary_matrix(PRIMARIES_BMD_FILM_V1,
                                                       WHITEPOINT_BMD_FILM_V1))
"""
*BMD Film V1* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BMD_FILM_V1_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BMD_FILM_V1 = (np.linalg.inv(MATRIX_BMD_FILM_V1_TO_XYZ))
"""
*CIE XYZ* tristimulus values to *BMD Film V1* colourspace matrix.

MATRIX_XYZ_TO_BMD_FILM_V1 : array_like, (3, 3)
"""

RGB_COLOURSPACE_BMD_FILM_V1 = RGB_Colourspace(
    'BMD Film V1',
    PRIMARIES_BMD_FILM_V1,
    WHITEPOINT_BMD_FILM_V1,
    WHITEPOINT_NAME_BMD_FILM_V1,
    MATRIX_BMD_FILM_V1_TO_XYZ,
    MATRIX_XYZ_TO_BMD_FILM_V1,
    log_encoding_BMDFilm,
    log_decoding_BMDFilm,
)
RGB_COLOURSPACE_BMD_FILM_V1.__doc__ = """
*BMD Film V1* colourspace.

References
----------
:cite:`Blackmagic2020a`

RGB_COLOURSPACE_BMD_FILM_V1 : RGB_Colourspace
"""

PRIMARIES_BMD_4K_FILM_V1 = np.array([
    [0.7422, 0.2859],
    [0.4140, 1.3035],
    [0.0342, -0.0833],
])
"""
*BMD 4K Film V1* colourspace primaries.

PRIMARIES_BMD_4K_FILM_V1 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BMD_4K_FILM_V1 = 'BMD White'
"""
*BMD 4K Film V1* colourspace whitepoint name.

WHITEPOINT_BMD_4K_FILM_V1 : unicode
"""

WHITEPOINT_BMD_4K_FILM_V1 = np.array([0.3135, 0.3305])
"""
*BMD 4K Film V1* colourspace whitepoint.

WHITEPOINT_BMD_4K_FILM_V1 : ndarray
"""

MATRIX_BMD_4K_FILM_V1_TO_XYZ = (normalised_primary_matrix(
    PRIMARIES_BMD_4K_FILM_V1, WHITEPOINT_BMD_4K_FILM_V1))
"""
*BMD 4K Film V1* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BMD_4K_FILM_V1_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BMD_4K_FILM_V1 = (np.linalg.inv(MATRIX_BMD_4K_FILM_V1_TO_XYZ))
"""
*CIE XYZ* tristimulus values to *BMD 4K Film V1* colourspace matrix.

MATRIX_XYZ_TO_BMD_4K_FILM_V1 : array_like, (3, 3)
"""

RGB_COLOURSPACE_BMD_4K_FILM_V1 = RGB_Colourspace(
    'BMD 4K Film V1',
    PRIMARIES_BMD_4K_FILM_V1,
    WHITEPOINT_BMD_4K_FILM_V1,
    WHITEPOINT_NAME_BMD_4K_FILM_V1,
    MATRIX_BMD_4K_FILM_V1_TO_XYZ,
    MATRIX_XYZ_TO_BMD_4K_FILM_V1,
    log_encoding_BMD4KFilm,
    log_decoding_BMD4KFilm,
)
RGB_COLOURSPACE_BMD_4K_FILM_V1.__doc__ = """
*BMD 4K Film V1* colourspace.

References
----------
:cite:`Blackmagic2020a`

RGB_COLOURSPACE_BMD_4K_FILM_V1 : RGB_Colourspace
"""

PRIMARIES_BMD_4K_FILM_V3 = np.array([
    [1.0625, 0.3948],
    [0.3689, 0.7775],
    [0.0956, -0.0332],
])
"""
*BMD 4K Film V3* colourspace primaries.

PRIMARIES_BMD_4K_FILM_V3 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BMD_4K_FILM_V3 = 'BMD White'
"""
*BMD 4K Film V3* colourspace whitepoint name.

WHITEPOINT_BMD_4K_FILM_V3 : unicode
"""

WHITEPOINT_BMD_4K_FILM_V3 = np.array([0.3135, 0.3305])
"""
*BMD 4K Film V3* colourspace whitepoint.

WHITEPOINT_BMD_4K_FILM_V3 : ndarray
"""

MATRIX_BMD_4K_FILM_V3_TO_XYZ = (normalised_primary_matrix(
    PRIMARIES_BMD_4K_FILM_V3, WHITEPOINT_BMD_4K_FILM_V3))
"""
*BMD 4K Film V3* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BMD_4K_FILM_V3_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BMD_4K_FILM_V3 = (np.linalg.inv(MATRIX_BMD_4K_FILM_V3_TO_XYZ))
"""
*CIE XYZ* tristimulus values to *BMD 4K Film V3* colourspace matrix.

MATRIX_XYZ_TO_BMD_4K_FILM_V3 : array_like, (3, 3)
"""

RGB_COLOURSPACE_BMD_4K_FILM_V3 = RGB_Colourspace(
    'BMD 4K Film V3',
    PRIMARIES_BMD_4K_FILM_V3,
    WHITEPOINT_BMD_4K_FILM_V3,
    WHITEPOINT_NAME_BMD_4K_FILM_V3,
    MATRIX_BMD_4K_FILM_V3_TO_XYZ,
    MATRIX_XYZ_TO_BMD_4K_FILM_V3,
    log_encoding_BMD4KFilm,
    log_decoding_BMD4KFilm,
)
RGB_COLOURSPACE_BMD_4K_FILM_V3.__doc__ = """
*BMD 4K Film V3* colourspace.

References
----------
:cite:`Blackmagic2020a`

RGB_COLOURSPACE_BMD_4K_FILM_V3 : RGB_Colourspace
"""

PRIMARIES_BMD_46K_FILM_V1 = np.array([
    [0.9175, 0.2983],
    [0.2983, 1.2835],
    [0.0756, -0.0860],
])
"""
*BMD 46K Film V1* colourspace primaries.

PRIMARIES_BMD_46K_FILM_V1 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BMD_46K_FILM_V1 = 'D65'
"""
*BMD 46K Film V1* colourspace whitepoint name.

WHITEPOINT_BMD_46K_FILM_V1 : unicode
"""

WHITEPOINT_BMD_46K_FILM_V1 = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BMD_46K_FILM_V1])
"""
*BMD 46K Film V1* colourspace whitepoint.

WHITEPOINT_BMD_46K_FILM_V1 : ndarray
"""

MATRIX_BMD_46K_FILM_V1_TO_XYZ = (normalised_primary_matrix(
    PRIMARIES_BMD_46K_FILM_V1, WHITEPOINT_BMD_46K_FILM_V1))
"""
*BMD 46K Film V1* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BMD_46K_FILM_V1_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BMD_46K_FILM_V1 = (np.linalg.inv(MATRIX_BMD_46K_FILM_V1_TO_XYZ))
"""
*CIE XYZ* tristimulus values to *BMD 46K Film V1* colourspace matrix.

MATRIX_XYZ_TO_BMD_46K_FILM_V1 : array_like, (3, 3)
"""

RGB_COLOURSPACE_BMD_46K_FILM_V1 = RGB_Colourspace(
    'BMD 46K Film V1',
    PRIMARIES_BMD_46K_FILM_V1,
    WHITEPOINT_BMD_46K_FILM_V1,
    WHITEPOINT_NAME_BMD_46K_FILM_V1,
    MATRIX_BMD_46K_FILM_V1_TO_XYZ,
    MATRIX_XYZ_TO_BMD_46K_FILM_V1,
    log_encoding_BMD46KFilm,
    log_decoding_BMD46KFilm,
)
RGB_COLOURSPACE_BMD_46K_FILM_V1.__doc__ = """
*BMD 46K Film V1* colourspace.

References
----------
:cite:`Blackmagic2020a`

RGB_COLOURSPACE_BMD_46K_FILM_V1 : RGB_Colourspace
"""

PRIMARIES_BMD_46K_FILM_V3 = np.array([
    [0.8608, 0.3689],
    [0.3282, 0.6156],
    [0.0783, -0.0233],
])
"""
*BMD 46K Film V3* colourspace primaries.

PRIMARIES_BMD_46K_FILM_V3 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BMD_46K_FILM_V3 = 'D65'
"""
*BMD 46K Film V3* colourspace whitepoint name.

WHITEPOINT_BMD_46K_FILM_V3 : unicode
"""

WHITEPOINT_BMD_46K_FILM_V3 = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BMD_46K_FILM_V3])
"""
*BMD 46K Film V3* colourspace whitepoint.

WHITEPOINT_BMD_46K_FILM_V3 : ndarray
"""

MATRIX_BMD_46K_FILM_V3_TO_XYZ = (normalised_primary_matrix(
    PRIMARIES_BMD_46K_FILM_V3, WHITEPOINT_BMD_46K_FILM_V3))
"""
*BMD 46K Film V3* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BMD_46K_FILM_V3_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BMD_46K_FILM_V3 = (np.linalg.inv(MATRIX_BMD_46K_FILM_V3_TO_XYZ))
"""
*CIE XYZ* tristimulus values to *BMD 46K Film V3* colourspace matrix.

MATRIX_XYZ_TO_BMD_46K_FILM_V3 : array_like, (3, 3)
"""

RGB_COLOURSPACE_BMD_46K_FILM_V3 = RGB_Colourspace(
    'BMD 46K Film V3',
    PRIMARIES_BMD_46K_FILM_V3,
    WHITEPOINT_BMD_46K_FILM_V3,
    WHITEPOINT_NAME_BMD_46K_FILM_V3,
    MATRIX_BMD_46K_FILM_V3_TO_XYZ,
    MATRIX_XYZ_TO_BMD_46K_FILM_V3,
    log_encoding_BMD46KFilm,
    log_decoding_BMD46KFilm,
)
RGB_COLOURSPACE_BMD_46K_FILM_V3.__doc__ = """
*BMD 46K Film V3* colourspace.

References
----------
:cite:`Blackmagic2020a`

RGB_COLOURSPACE_BMD_46K_FILM_V3 : RGB_Colourspace
"""

PRIMARIES_BMD_WIDE_GAMUT_V4 = np.array([
    [0.7177, 0.3171],
    [0.2280, 0.8616],
    [0.1006, -0.0820],
])
"""
*BMD Wide Gamut V4* colourspace primaries.

PRIMARIES_BMD_WIDE_GAMUT_V4 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BMD_WIDE_GAMUT_V4 = 'D65'
"""
*BMD Wide Gamut V4* colourspace whitepoint name.

WHITEPOINT_BMD_WIDE_GAMUT_V4 : unicode
"""

WHITEPOINT_BMD_WIDE_GAMUT_V4 = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BMD_WIDE_GAMUT_V4])
"""
*BMD Wide Gamut V4* colourspace whitepoint.

WHITEPOINT_BMD_WIDE_GAMUT_V4 : ndarray
"""

MATRIX_BMD_WIDE_GAMUT_V4_TO_XYZ = (normalised_primary_matrix(
    PRIMARIES_BMD_WIDE_GAMUT_V4, WHITEPOINT_BMD_WIDE_GAMUT_V4))
"""
*BMD Wide Gamut V4* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BMD_WIDE_GAMUT_V4_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BMD_WIDE_GAMUT_V4 = (
    np.linalg.inv(MATRIX_BMD_WIDE_GAMUT_V4_TO_XYZ))
"""
*CIE XYZ* tristimulus values to *BMD Wide Gamut V4* colourspace matrix.

MATRIX_XYZ_TO_BMD_WIDE_GAMUT_V4 : array_like, (3, 3)
"""

RGB_COLOURSPACE_BMD_WIDE_GAMUT_V4 = RGB_Colourspace(
    'BMD Wide Gamut V4',
    PRIMARIES_BMD_WIDE_GAMUT_V4,
    WHITEPOINT_BMD_WIDE_GAMUT_V4,
    WHITEPOINT_NAME_BMD_WIDE_GAMUT_V4,
    MATRIX_BMD_WIDE_GAMUT_V4_TO_XYZ,
    MATRIX_XYZ_TO_BMD_WIDE_GAMUT_V4,
    log_encoding_BMDPocket4KFilmV4,
    log_decoding_BMDPocket4KFilmV4,
)
RGB_COLOURSPACE_BMD_WIDE_GAMUT_V4.__doc__ = """
*BMD Wide Gamut V4* colourspace.

References
----------
:cite:`Blackmagic2020a`

RGB_COLOURSPACE_BMD_WIDE_GAMUT_V4 : RGB_Colourspace
"""
