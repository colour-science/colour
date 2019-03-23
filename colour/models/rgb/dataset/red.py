# -*- coding: utf-8 -*-
"""
RED Colourspaces
================

Defines the *RED* colourspaces:

-   :attr:`colour.models.RED_COLOR_COLOURSPACE`
-   :attr:`colour.models.RED_COLOR_2_COLOURSPACE`
-   :attr:`colour.models.RED_COLOR_3_COLOURSPACE`
-   :attr:`colour.models.RED_COLOR_4_COLOURSPACE`
-   :attr:`colour.models.DRAGON_COLOR_COLOURSPACE`
-   :attr:`colour.models.DRAGON_COLOR_2_COLOURSPACE`
-   :attr:`colour.models.RED_WIDE_GAMUT_RGB_COLOURSPACE`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Mansencal2015d` : Mansencal, T. (2015). RED Colourspaces Derivation.
    Retrieved May 20, 2015, from
    https://www.colour-science.org/posts/red-colourspaces-derivation
-   :cite:`Nattress2016a` : Nattress, G. (2016). Private Discussion with
    Shaw, N.
-   :cite:`SonyImageworks2012a` : Sony Imageworks. (2012). make.py. Retrieved
    November 27, 2014, from https://github.com/imageworks/OpenColorIO-Configs/\
blob/master/nuke-default/make.py
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace, normalised_primary_matrix, log_encoding_REDLogFilm,
    log_decoding_REDLogFilm, log_encoding_Log3G10, log_decoding_Log3G10)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RED_COLOR_PRIMARIES', 'RED_COLOR_WHITEPOINT_NAME', 'RED_COLOR_WHITEPOINT',
    'RED_COLOR_TO_XYZ_MATRIX', 'XYZ_TO_RED_COLOR_MATRIX',
    'RED_COLOR_COLOURSPACE', 'RED_COLOR_2_PRIMARIES',
    'RED_COLOR_2_WHITEPOINT_NAME', 'RED_COLOR_2_WHITEPOINT',
    'RED_COLOR_2_TO_XYZ_MATRIX', 'XYZ_TO_RED_COLOR_2_MATRIX',
    'RED_COLOR_2_COLOURSPACE', 'RED_COLOR_3_PRIMARIES',
    'RED_COLOR_3_WHITEPOINT_NAME', 'RED_COLOR_3_WHITEPOINT',
    'RED_COLOR_3_TO_XYZ_MATRIX', 'XYZ_TO_RED_COLOR_3_MATRIX',
    'RED_COLOR_3_COLOURSPACE', 'RED_COLOR_4_PRIMARIES',
    'RED_COLOR_4_WHITEPOINT_NAME', 'RED_COLOR_4_WHITEPOINT',
    'RED_COLOR_4_TO_XYZ_MATRIX', 'XYZ_TO_RED_COLOR_4_MATRIX',
    'RED_COLOR_4_COLOURSPACE', 'DRAGON_COLOR_PRIMARIES',
    'DRAGON_COLOR_WHITEPOINT_NAME', 'DRAGON_COLOR_WHITEPOINT',
    'DRAGON_COLOR_TO_XYZ_MATRIX', 'XYZ_TO_DRAGON_COLOR_MATRIX',
    'DRAGON_COLOR_COLOURSPACE', 'DRAGON_COLOR_2_PRIMARIES',
    'DRAGON_COLOR_2_WHITEPOINT_NAME', 'DRAGON_COLOR_2_WHITEPOINT',
    'DRAGON_COLOR_2_TO_XYZ_MATRIX', 'XYZ_TO_DRAGON_COLOR_2_MATRIX',
    'DRAGON_COLOR_2_COLOURSPACE', 'RED_WIDE_GAMUT_RGB_PRIMARIES',
    'RED_WIDE_GAMUT_RGB_WHITEPOINT_NAME', 'RED_WIDE_GAMUT_RGB_WHITEPOINT',
    'RED_WIDE_GAMUT_RGB_TO_XYZ_MATRIX', 'XYZ_TO_RED_WIDE_GAMUT_RGB_MATRIX',
    'RED_WIDE_GAMUT_RGB_COLOURSPACE'
]

RED_COLOR_PRIMARIES = np.array([
    [0.701058563171395, 0.330180975940326],
    [0.298811317306316, 0.625169245953133],
    [0.135038675201355, 0.035261776551191],
])
"""
*REDcolor* colourspace primaries.

RED_COLOR_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_WHITEPOINT_NAME = 'D65'
"""
*REDcolor* colourspace whitepoint name.

RED_COLOR_WHITEPOINT_NAME : unicode
"""

RED_COLOR_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    RED_COLOR_WHITEPOINT_NAME])
"""
*REDcolor* colourspace whitepoint.

RED_COLOR_WHITEPOINT : ndarray
"""

RED_COLOR_TO_XYZ_MATRIX = normalised_primary_matrix(RED_COLOR_PRIMARIES,
                                                    RED_COLOR_WHITEPOINT)
"""
*REDcolor* colourspace to *CIE XYZ* tristimulus values matrix.

RED_COLOR_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_RED_COLOR_MATRIX = np.linalg.inv(RED_COLOR_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *REDcolor* colourspace matrix.

XYZ_TO_RED_COLOR_MATRIX : array_like, (3, 3)
"""

RED_COLOR_COLOURSPACE = RGB_Colourspace(
    'REDcolor',
    RED_COLOR_PRIMARIES,
    RED_COLOR_WHITEPOINT,
    RED_COLOR_WHITEPOINT_NAME,
    RED_COLOR_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RED_COLOR_COLOURSPACE.__doc__ = """
*REDcolor* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`

RED_COLOR_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_2_PRIMARIES = np.array([
    [0.897407221929776, 0.330776225980398],
    [0.296022094516625, 0.684635550900945],
    [0.099799512883393, -0.023000513177992],
])
"""
*REDcolor2* colourspace primaries.

RED_COLOR_2_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_2_WHITEPOINT_NAME = RED_COLOR_WHITEPOINT_NAME
"""
*REDcolor2* colourspace whitepoint name.

RED_COLOR_2_WHITEPOINT_NAME : unicode
"""

RED_COLOR_2_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*REDcolor2* colourspace whitepoint.

RED_COLOR_2_WHITEPOINT : ndarray
"""

RED_COLOR_2_TO_XYZ_MATRIX = normalised_primary_matrix(RED_COLOR_2_PRIMARIES,
                                                      RED_COLOR_2_WHITEPOINT)
"""
*REDcolor2* colourspace to *CIE XYZ* tristimulus values matrix.

RED_COLOR_2_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_RED_COLOR_2_MATRIX = np.linalg.inv(RED_COLOR_2_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *REDcolor2* colourspace matrix.

XYZ_TO_RED_COLOR_2_MATRIX : array_like, (3, 3)
"""

RED_COLOR_2_COLOURSPACE = RGB_Colourspace(
    'REDcolor2',
    RED_COLOR_2_PRIMARIES,
    RED_COLOR_2_WHITEPOINT,
    RED_COLOR_2_WHITEPOINT_NAME,
    RED_COLOR_2_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_2_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RED_COLOR_2_COLOURSPACE.__doc__ = """
*REDcolor2* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`

RED_COLOR_2_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_3_PRIMARIES = np.array([
    [0.702598658589917, 0.330185588938484],
    [0.295782235737268, 0.689748258397534],
    [0.111090529079787, -0.004332320984771],
])
"""
*REDcolor3* colourspace primaries.

RED_COLOR_3_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_3_WHITEPOINT_NAME = RED_COLOR_WHITEPOINT_NAME
"""
*REDcolor3* colourspace whitepoint name.

RED_COLOR_3_WHITEPOINT_NAME : unicode
"""

RED_COLOR_3_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*REDcolor3* colourspace whitepoint.

RED_COLOR_3_WHITEPOINT : ndarray
"""

RED_COLOR_3_TO_XYZ_MATRIX = normalised_primary_matrix(RED_COLOR_3_PRIMARIES,
                                                      RED_COLOR_3_WHITEPOINT)
"""
*REDcolor3* colourspace to *CIE XYZ* tristimulus values matrix.

RED_COLOR_3_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_RED_COLOR_3_MATRIX = np.linalg.inv(RED_COLOR_3_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *REDcolor3* colourspace matrix.

XYZ_TO_RED_COLOR_3_MATRIX : array_like, (3, 3)
"""

RED_COLOR_3_COLOURSPACE = RGB_Colourspace(
    'REDcolor3',
    RED_COLOR_3_PRIMARIES,
    RED_COLOR_3_WHITEPOINT,
    RED_COLOR_3_WHITEPOINT_NAME,
    RED_COLOR_3_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_3_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RED_COLOR_3_COLOURSPACE.__doc__ = """
*REDcolor3* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`

RED_COLOR_3_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_4_PRIMARIES = np.array([
    [0.702598154635438, 0.330185096210515],
    [0.295782328047083, 0.689748253964859],
    [0.144459236489795, 0.050837720977386],
])
"""
*REDcolor4* colourspace primaries.

RED_COLOR_4_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_4_WHITEPOINT_NAME = RED_COLOR_WHITEPOINT_NAME
"""
*REDcolor4* colourspace whitepoint name.

RED_COLOR_4_WHITEPOINT_NAME : unicode
"""

RED_COLOR_4_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*REDcolor4* colourspace whitepoint.

RED_COLOR_4_WHITEPOINT : ndarray
"""

RED_COLOR_4_TO_XYZ_MATRIX = normalised_primary_matrix(RED_COLOR_4_PRIMARIES,
                                                      RED_COLOR_4_WHITEPOINT)
"""
*REDcolor4* colourspace to *CIE XYZ* tristimulus values matrix.

RED_COLOR_4_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_RED_COLOR_4_MATRIX = np.linalg.inv(RED_COLOR_4_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *REDcolor4* colourspace matrix.

XYZ_TO_RED_COLOR_4_MATRIX : array_like, (3, 3)
"""

RED_COLOR_4_COLOURSPACE = RGB_Colourspace(
    'REDcolor4',
    RED_COLOR_4_PRIMARIES,
    RED_COLOR_4_WHITEPOINT,
    RED_COLOR_4_WHITEPOINT_NAME,
    RED_COLOR_4_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_4_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
RED_COLOR_4_COLOURSPACE.__doc__ = """
*REDcolor4* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`

RED_COLOR_4_COLOURSPACE : RGB_Colourspace
"""

DRAGON_COLOR_PRIMARIES = np.array([
    [0.758655892599321, 0.330355348611293],
    [0.294923619810175, 0.708053242065117],
    [0.085961601167585, -0.045879436983969],
])
"""
*DRAGONcolor* colourspace primaries.

DRAGON_COLOR_PRIMARIES : ndarray, (3, 2)
"""

DRAGON_COLOR_WHITEPOINT_NAME = RED_COLOR_WHITEPOINT_NAME
"""
*DRAGONcolor* colourspace whitepoint name.

DRAGON_COLOR_WHITEPOINT_NAME : unicode
"""

DRAGON_COLOR_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*DRAGONcolor* colourspace whitepoint.

DRAGON_COLOR_WHITEPOINT : ndarray
"""

DRAGON_COLOR_TO_XYZ_MATRIX = normalised_primary_matrix(
    DRAGON_COLOR_PRIMARIES, DRAGON_COLOR_WHITEPOINT)
"""
*DRAGONcolor* colourspace to *CIE XYZ* tristimulus values matrix.

DRAGON_COLOR_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DRAGON_COLOR_MATRIX = np.linalg.inv(DRAGON_COLOR_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *DRAGONcolor* colourspace matrix.

XYZ_TO_DRAGON_COLOR_MATRIX : array_like, (3, 3)
"""

DRAGON_COLOR_COLOURSPACE = RGB_Colourspace(
    'DRAGONcolor',
    DRAGON_COLOR_PRIMARIES,
    DRAGON_COLOR_WHITEPOINT,
    DRAGON_COLOR_WHITEPOINT_NAME,
    DRAGON_COLOR_TO_XYZ_MATRIX,
    XYZ_TO_DRAGON_COLOR_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
DRAGON_COLOR_COLOURSPACE.__doc__ = """
*DRAGONcolor* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`

DRAGON_COLOR_COLOURSPACE : RGB_Colourspace
"""

DRAGON_COLOR_2_PRIMARIES = np.array([
    [0.758656214177604, 0.330355835762678],
    [0.294923887732982, 0.708053363192126],
    [0.144168726866337, 0.050357384587121],
])
"""
*DRAGONcolor2* colourspace primaries.

DRAGON_COLOR_2_PRIMARIES : ndarray, (3, 2)
"""

DRAGON_COLOR_2_WHITEPOINT_NAME = RED_COLOR_WHITEPOINT_NAME
"""
*DRAGONcolor2* colourspace whitepoint name.

DRAGON_COLOR_2_WHITEPOINT_NAME : unicode
"""

DRAGON_COLOR_2_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*DRAGONcolor2* colourspace whitepoint.

DRAGON_COLOR_2_WHITEPOINT : ndarray
"""

DRAGON_COLOR_2_TO_XYZ_MATRIX = normalised_primary_matrix(
    DRAGON_COLOR_2_PRIMARIES, DRAGON_COLOR_2_WHITEPOINT)
"""
*DRAGONcolor2* colourspace to *CIE XYZ* tristimulus values matrix.

DRAGON_COLOR_2_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DRAGON_COLOR_2_MATRIX = np.linalg.inv(DRAGON_COLOR_2_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *DRAGONcolor2* colourspace matrix.

XYZ_TO_DRAGON_COLOR_2_MATRIX : array_like, (3, 3)
"""

DRAGON_COLOR_2_COLOURSPACE = RGB_Colourspace(
    'DRAGONcolor2',
    DRAGON_COLOR_2_PRIMARIES,
    DRAGON_COLOR_2_WHITEPOINT,
    DRAGON_COLOR_2_WHITEPOINT_NAME,
    DRAGON_COLOR_2_TO_XYZ_MATRIX,
    XYZ_TO_DRAGON_COLOR_2_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
)
DRAGON_COLOR_2_COLOURSPACE.__doc__ = """
*DRAGONcolor2* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`SonyImageworks2012a`

DRAGON_COLOR_2_COLOURSPACE : RGB_Colourspace
"""

RED_WIDE_GAMUT_RGB_PRIMARIES = np.array([
    [0.780308, 0.304253],
    [0.121595, 1.493994],
    [0.095612, -0.084589],
])
"""
*REDWideGamutRGB* colourspace primaries.

RED_WIDE_GAMUT_RGB_PRIMARIES : ndarray, (3, 2)
"""

RED_WIDE_GAMUT_RGB_WHITEPOINT_NAME = RED_COLOR_WHITEPOINT_NAME
"""
*REDWideGamutRGB* colourspace whitepoint name.

RED_WIDE_GAMUT_RGB_WHITEPOINT_NAME : unicode
"""

RED_WIDE_GAMUT_RGB_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*REDWideGamutRGB* colourspace whitepoint.

RED_WIDE_GAMUT_RGB_WHITEPOINT : ndarray
"""

RED_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = np.array([
    [0.735275, 0.068609, 0.146571],
    [0.286694, 0.842979, -0.129673],
    [-0.079681, -0.347343, 1.516082],
])
"""
*REDWideGamutRGB* colourspace to *CIE XYZ* tristimulus values matrix.

RED_WIDE_GAMUT_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_RED_WIDE_GAMUT_RGB_MATRIX = np.linalg.inv(
    RED_WIDE_GAMUT_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *REDWideGamutRGB* colourspace matrix.

XYZ_TO_RED_WIDE_GAMUT_RGB_MATRIX : array_like, (3, 3)
"""

RED_WIDE_GAMUT_RGB_COLOURSPACE = RGB_Colourspace(
    'REDWideGamutRGB',
    RED_WIDE_GAMUT_RGB_PRIMARIES,
    RED_WIDE_GAMUT_RGB_WHITEPOINT,
    RED_WIDE_GAMUT_RGB_WHITEPOINT_NAME,
    RED_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
    XYZ_TO_RED_WIDE_GAMUT_RGB_MATRIX,
    log_encoding_Log3G10,
    log_decoding_Log3G10,
)
RED_WIDE_GAMUT_RGB_COLOURSPACE.__doc__ = """
*REDWideGamutRGB* colourspace.

References
----------
:cite:`Mansencal2015d`, :cite:`Nattress2016a`, :cite:`SonyImageworks2012a`

RED_WIDE_GAMUT_RGB_COLOURSPACE : RGB_Colourspace
"""
