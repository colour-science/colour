# -*- coding: utf-8 -*-
"""
RED Colourspaces
================

Defines the *RED* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_RED_COLOR`
-   :attr:`colour.models.RGB_COLOURSPACE_RED_COLOR_2`
-   :attr:`colour.models.RGB_COLOURSPACE_RED_COLOR_3`
-   :attr:`colour.models.RGB_COLOURSPACE_RED_COLOR_4`
-   :attr:`colour.models.RGB_COLOURSPACE_DRAGON_COLOR`
-   :attr:`colour.models.RGB_COLOURSPACE_DRAGON_COLOR_2`
-   :attr:`colour.models.RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB`

References
----------
-   :cite:`Mansencal2015d` : Mansencal, T. (2015). RED Colourspaces Derivation.
    Retrieved May 20, 2015, from
    https://www.colour-science.org/posts/red-colourspaces-derivation
-   :cite:`Nattress2016a` : Nattress, G. (2016). Private Discussion with Shaw,
    N.
-   :cite:`SonyImageworks2012a` : Sony Imageworks. (2012). make.py. Retrieved
    November 27, 2014, from
    https://github.com/imageworks/OpenColorIO-Configs/blob/master/\
nuke-default/make.py
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    normalised_primary_matrix,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
    log_encoding_Log3G10,
    log_decoding_Log3G10,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_RED_COLOR',
    'WHITEPOINT_NAME_RED_COLOR',
    'CCS_WHITEPOINT_RED_COLOR',
    'MATRIX_RED_COLOR_TO_XYZ',
    'MATRIX_XYZ_TO_RED_COLOR',
    'RGB_COLOURSPACE_RED_COLOR',
    'PRIMARIES_RED_COLOR_2',
    'WHITEPOINT_NAME_RED_COLOR_2',
    'CCS_WHITEPOINT_RED_COLOR_2',
    'MATRIX_RED_COLOR_2_TO_XYZ',
    'MATRIX_XYZ_TO_RED_COLOR_2',
    'RGB_COLOURSPACE_RED_COLOR_2',
    'PRIMARIES_RED_COLOR_3',
    'WHITEPOINT_NAME_RED_COLOR_3',
    'CCS_WHITEPOINT_RED_COLOR_3',
    'MATRIX_RED_COLOR_3_TO_XYZ',
    'MATRIX_XYZ_TO_RED_COLOR_3',
    'RGB_COLOURSPACE_RED_COLOR_3',
    'PRIMARIES_RED_COLOR_4',
    'WHITEPOINT_NAME_RED_COLOR_4',
    'CCS_WHITEPOINT_RED_COLOR_4',
    'MATRIX_RED_COLOR_4_TO_XYZ',
    'MATRIX_XYZ_TO_RED_COLOR_4',
    'RGB_COLOURSPACE_RED_COLOR_4',
    'PRIMARIES_DRAGON_COLOR',
    'WHITEPOINT_NAME_DRAGON_COLOR',
    'CCS_WHITEPOINT_DRAGON_COLOR',
    'MATRIX_DRAGON_COLOR_TO_XYZ',
    'MATRIX_XYZ_TO_DRAGON_COLOR',
    'RGB_COLOURSPACE_DRAGON_COLOR',
    'PRIMARIES_DRAGON_COLOR_2',
    'WHITEPOINT_NAME_DRAGON_COLOR_2',
    'CCS_WHITEPOINT_DRAGON_COLOR_2',
    'MATRIX_DRAGON_COLOR_2_TO_XYZ',
    'MATRIX_XYZ_TO_DRAGON_COLOR_2',
    'RGB_COLOURSPACE_DRAGON_COLOR_2',
    'PRIMARIES_RED_WIDE_GAMUT_RGB',
    'WHITEPOINT_NAME_RED_WIDE_GAMUT_RGB',
    'CCS_WHITEPOINT_RED_WIDE_GAMUT_RGB',
    'MATRIX_RED_WIDE_GAMUT_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_RED_WIDE_GAMUT_RGB',
    'RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB',
]

PRIMARIES_RED_COLOR: NDArray = np.array([
    [0.701058563171395, 0.330180975940326],
    [0.298811317306316, 0.625169245953133],
    [0.135038675201355, 0.035261776551191],
])
"""
*REDcolor* colourspace primaries.
"""

WHITEPOINT_NAME_RED_COLOR: str = 'D65'
"""
*REDcolor* colourspace whitepoint name.
"""

CCS_WHITEPOINT_RED_COLOR: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_RED_COLOR])
"""
*REDcolor* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_RED_COLOR_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_RED_COLOR, CCS_WHITEPOINT_RED_COLOR)
"""
*REDcolor* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_RED_COLOR: NDArray = np.linalg.inv(MATRIX_RED_COLOR_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *REDcolor* colourspace matrix.
"""

RGB_COLOURSPACE_RED_COLOR: RGB_Colourspace = RGB_Colourspace(
    'REDcolor',
    PRIMARIES_RED_COLOR,
    CCS_WHITEPOINT_RED_COLOR,
    WHITEPOINT_NAME_RED_COLOR,
    MATRIX_RED_COLOR_TO_XYZ,
    MATRIX_XYZ_TO_RED_COLOR,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RGB_COLOURSPACE_RED_COLOR.__doc__ = """
*REDcolor* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`
"""

PRIMARIES_RED_COLOR_2: NDArray = np.array([
    [0.897407221929776, 0.330776225980398],
    [0.296022094516625, 0.684635550900945],
    [0.099799512883393, -0.023000513177992],
])
"""
*REDcolor2* colourspace primaries.
"""

WHITEPOINT_NAME_RED_COLOR_2: str = WHITEPOINT_NAME_RED_COLOR
"""
*REDcolor2* colourspace whitepoint name.
"""

CCS_WHITEPOINT_RED_COLOR_2: NDArray = CCS_WHITEPOINT_RED_COLOR
"""
*REDcolor2* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_RED_COLOR_2_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_RED_COLOR_2, CCS_WHITEPOINT_RED_COLOR_2)
"""
*REDcolor2* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_RED_COLOR_2: NDArray = np.linalg.inv(MATRIX_RED_COLOR_2_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *REDcolor2* colourspace matrix.
"""

RGB_COLOURSPACE_RED_COLOR_2: RGB_Colourspace = RGB_Colourspace(
    'REDcolor2',
    PRIMARIES_RED_COLOR_2,
    CCS_WHITEPOINT_RED_COLOR_2,
    WHITEPOINT_NAME_RED_COLOR_2,
    MATRIX_RED_COLOR_2_TO_XYZ,
    MATRIX_XYZ_TO_RED_COLOR_2,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RGB_COLOURSPACE_RED_COLOR_2.__doc__ = """
*REDcolor2* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`
"""

PRIMARIES_RED_COLOR_3: NDArray = np.array([
    [0.702598658589917, 0.330185588938484],
    [0.295782235737268, 0.689748258397534],
    [0.111090529079787, -0.004332320984771],
])
"""
*REDcolor3* colourspace primaries.
"""

WHITEPOINT_NAME_RED_COLOR_3: str = WHITEPOINT_NAME_RED_COLOR
"""
*REDcolor3* colourspace whitepoint name.
"""

CCS_WHITEPOINT_RED_COLOR_3: NDArray = CCS_WHITEPOINT_RED_COLOR
"""
*REDcolor3* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_RED_COLOR_3_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_RED_COLOR_3, CCS_WHITEPOINT_RED_COLOR_3)
"""
*REDcolor3* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_RED_COLOR_3: NDArray = np.linalg.inv(MATRIX_RED_COLOR_3_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *REDcolor3* colourspace matrix.
"""

RGB_COLOURSPACE_RED_COLOR_3: RGB_Colourspace = RGB_Colourspace(
    'REDcolor3',
    PRIMARIES_RED_COLOR_3,
    CCS_WHITEPOINT_RED_COLOR_3,
    WHITEPOINT_NAME_RED_COLOR_3,
    MATRIX_RED_COLOR_3_TO_XYZ,
    MATRIX_XYZ_TO_RED_COLOR_3,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RGB_COLOURSPACE_RED_COLOR_3.__doc__ = """
*REDcolor3* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`
"""

PRIMARIES_RED_COLOR_4: NDArray = np.array([
    [0.702598154635438, 0.330185096210515],
    [0.295782328047083, 0.689748253964859],
    [0.144459236489795, 0.050837720977386],
])
"""
*REDcolor4* colourspace primaries.
"""

WHITEPOINT_NAME_RED_COLOR_4: str = WHITEPOINT_NAME_RED_COLOR
"""
*REDcolor4* colourspace whitepoint name.
"""

CCS_WHITEPOINT_RED_COLOR_4: NDArray = CCS_WHITEPOINT_RED_COLOR
"""
*REDcolor4* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_RED_COLOR_4_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_RED_COLOR_4, CCS_WHITEPOINT_RED_COLOR_4)
"""
*REDcolor4* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_RED_COLOR_4: NDArray = np.linalg.inv(MATRIX_RED_COLOR_4_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *REDcolor4* colourspace matrix.
"""

RGB_COLOURSPACE_RED_COLOR_4: RGB_Colourspace = RGB_Colourspace(
    'REDcolor4',
    PRIMARIES_RED_COLOR_4,
    CCS_WHITEPOINT_RED_COLOR_4,
    WHITEPOINT_NAME_RED_COLOR_4,
    MATRIX_RED_COLOR_4_TO_XYZ,
    MATRIX_XYZ_TO_RED_COLOR_4,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RGB_COLOURSPACE_RED_COLOR_4.__doc__ = """
*REDcolor4* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`
"""

PRIMARIES_DRAGON_COLOR: NDArray = np.array([
    [0.758655892599321, 0.330355348611293],
    [0.294923619810175, 0.708053242065117],
    [0.085961601167585, -0.045879436983969],
])
"""
*DRAGONcolor* colourspace primaries.
"""

WHITEPOINT_NAME_DRAGON_COLOR: str = WHITEPOINT_NAME_RED_COLOR
"""
*DRAGONcolor* colourspace whitepoint name.
"""

CCS_WHITEPOINT_DRAGON_COLOR: NDArray = CCS_WHITEPOINT_RED_COLOR
"""
*DRAGONcolor* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_DRAGON_COLOR_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_DRAGON_COLOR, CCS_WHITEPOINT_DRAGON_COLOR)
"""
*DRAGONcolor* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_DRAGON_COLOR: NDArray = np.linalg.inv(MATRIX_DRAGON_COLOR_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *DRAGONcolor* colourspace matrix.
"""

RGB_COLOURSPACE_DRAGON_COLOR: RGB_Colourspace = RGB_Colourspace(
    'DRAGONcolor',
    PRIMARIES_DRAGON_COLOR,
    CCS_WHITEPOINT_DRAGON_COLOR,
    WHITEPOINT_NAME_DRAGON_COLOR,
    MATRIX_DRAGON_COLOR_TO_XYZ,
    MATRIX_XYZ_TO_DRAGON_COLOR,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RGB_COLOURSPACE_DRAGON_COLOR.__doc__ = """
*DRAGONcolor* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`
"""

PRIMARIES_DRAGON_COLOR_2: NDArray = np.array([
    [0.758656214177604, 0.330355835762678],
    [0.294923887732982, 0.708053363192126],
    [0.144168726866337, 0.050357384587121],
])
"""
*DRAGONcolor2* colourspace primaries.
"""

WHITEPOINT_NAME_DRAGON_COLOR_2: str = WHITEPOINT_NAME_RED_COLOR
"""
*DRAGONcolor2* colourspace whitepoint name.
"""

CCS_WHITEPOINT_DRAGON_COLOR_2: NDArray = CCS_WHITEPOINT_RED_COLOR
"""
*DRAGONcolor2* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_DRAGON_COLOR_2_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_DRAGON_COLOR_2, CCS_WHITEPOINT_DRAGON_COLOR_2)
"""
*DRAGONcolor2* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_DRAGON_COLOR_2: NDArray = np.linalg.inv(
    MATRIX_DRAGON_COLOR_2_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *DRAGONcolor2* colourspace matrix.
"""

RGB_COLOURSPACE_DRAGON_COLOR_2: RGB_Colourspace = RGB_Colourspace(
    'DRAGONcolor2',
    PRIMARIES_DRAGON_COLOR_2,
    CCS_WHITEPOINT_DRAGON_COLOR_2,
    WHITEPOINT_NAME_DRAGON_COLOR_2,
    MATRIX_DRAGON_COLOR_2_TO_XYZ,
    MATRIX_XYZ_TO_DRAGON_COLOR_2,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RGB_COLOURSPACE_DRAGON_COLOR_2.__doc__ = """
*DRAGONcolor2* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`
"""

PRIMARIES_RED_WIDE_GAMUT_RGB: NDArray = np.array([
    [0.780308, 0.304253],
    [0.121595, 1.493994],
    [0.095612, -0.084589],
])
"""
*REDWideGamutRGB* colourspace primaries.
"""

WHITEPOINT_NAME_RED_WIDE_GAMUT_RGB: str = WHITEPOINT_NAME_RED_COLOR
"""
*REDWideGamutRGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_RED_WIDE_GAMUT_RGB = CCS_WHITEPOINT_RED_COLOR
"""
*REDWideGamutRGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_RED_WIDE_GAMUT_RGB_TO_XYZ: NDArray = np.array([
    [0.735275, 0.068609, 0.146571],
    [0.286694, 0.842979, -0.129673],
    [-0.079681, -0.347343, 1.516082],
])
"""
*REDWideGamutRGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_RED_WIDE_GAMUT_RGB: NDArray = np.linalg.inv(
    MATRIX_RED_WIDE_GAMUT_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *REDWideGamutRGB* colourspace matrix.
"""

RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB: RGB_Colourspace = RGB_Colourspace(
    'REDWideGamutRGB',
    PRIMARIES_RED_WIDE_GAMUT_RGB,
    CCS_WHITEPOINT_RED_WIDE_GAMUT_RGB,
    WHITEPOINT_NAME_RED_WIDE_GAMUT_RGB,
    MATRIX_RED_WIDE_GAMUT_RGB_TO_XYZ,
    MATRIX_XYZ_TO_RED_WIDE_GAMUT_RGB,
    log_encoding_Log3G10,
    log_decoding_Log3G10,
)
RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB.__doc__ = """
*REDWideGamutRGB* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`Nattress2016a`, :cite:`SonyImageworks2012a`
"""
