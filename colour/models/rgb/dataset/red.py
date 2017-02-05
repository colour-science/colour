#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RED Colourspaces
================

Defines the *RED* colourspaces:

-   :attr:`RED_COLOR_COLOURSPACE`
-   :attr:`RED_COLOR_2_COLOURSPACE`
-   :attr:`RED_COLOR_3_COLOURSPACE`
-   :attr:`RED_COLOR_4_COLOURSPACE`
-   :attr:`DRAGON_COLOR_COLOURSPACE`
-   :attr:`DRAGON_COLOR_2_COLOURSPACE`
-   :attr:`RED_WIDE_GAMUT_RGB_COLOURSPACE`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Mansencal, T. (2015). RED Colourspaces Derivation. Retrieved May 20,
        2015, from http://colour-science.org/posts/red-colourspaces-derivation
.. [2]  Sony Imageworks. (2012). make.py. Retrieved November 27, 2014, from
        https://github.com/imageworks/OpenColorIO-Configs/\
blob/master/nuke-default/make.py
.. [3]  Nattress, G. (2016). Private Discussion with Shaw, N.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    normalised_primary_matrix,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
    log_encoding_Log3G10,
    log_decoding_Log3G10)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RED_COLOR_PRIMARIES',
           'RED_COLOR_ILLUMINANT',
           'RED_COLOR_WHITEPOINT',
           'RED_COLOR_TO_XYZ_MATRIX',
           'XYZ_TO_RED_COLOR_MATRIX',
           'RED_COLOR_COLOURSPACE',
           'RED_COLOR_2_PRIMARIES',
           'RED_COLOR_2_ILLUMINANT',
           'RED_COLOR_2_WHITEPOINT',
           'RED_COLOR_2_TO_XYZ_MATRIX',
           'XYZ_TO_RED_COLOR_2_MATRIX',
           'RED_COLOR_2_COLOURSPACE',
           'RED_COLOR_3_PRIMARIES',
           'RED_COLOR_3_ILLUMINANT',
           'RED_COLOR_3_WHITEPOINT',
           'RED_COLOR_3_TO_XYZ_MATRIX',
           'XYZ_TO_RED_COLOR_3_MATRIX',
           'RED_COLOR_3_COLOURSPACE',
           'RED_COLOR_4_PRIMARIES',
           'RED_COLOR_4_ILLUMINANT',
           'RED_COLOR_4_WHITEPOINT',
           'RED_COLOR_4_TO_XYZ_MATRIX',
           'XYZ_TO_RED_COLOR_4_MATRIX',
           'RED_COLOR_4_COLOURSPACE',
           'DRAGON_COLOR_PRIMARIES',
           'DRAGON_COLOR_ILLUMINANT',
           'DRAGON_COLOR_WHITEPOINT',
           'DRAGON_COLOR_TO_XYZ_MATRIX',
           'XYZ_TO_DRAGON_COLOR_MATRIX',
           'DRAGON_COLOR_COLOURSPACE',
           'DRAGON_COLOR_2_PRIMARIES',
           'DRAGON_COLOR_2_ILLUMINANT',
           'DRAGON_COLOR_2_WHITEPOINT',
           'DRAGON_COLOR_2_TO_XYZ_MATRIX',
           'XYZ_TO_DRAGON_COLOR_2_MATRIX',
           'DRAGON_COLOR_2_COLOURSPACE',
           'RED_WIDE_GAMUT_RGB_PRIMARIES',
           'RED_WIDE_GAMUT_RGB_ILLUMINANT',
           'RED_WIDE_GAMUT_RGB_WHITEPOINT',
           'RED_WIDE_GAMUT_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_RED_WIDE_GAMUT_RGB_MATRIX',
           'RED_WIDE_GAMUT_RGB_COLOURSPACE']

RED_COLOR_PRIMARIES = np.array(
    [[0.699747001290731, 0.329046930312637],
     [0.304264039023547, 0.623641145129115],
     [0.134913961296487, 0.034717441281345]])
"""
*REDcolor* colourspace primaries.

RED_COLOR_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_ILLUMINANT = 'D60'
"""
*REDcolor* colourspace whitepoint name as illuminant.

RED_COLOR_ILLUMINANT : unicode
"""

RED_COLOR_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][RED_COLOR_ILLUMINANT])
"""
*REDcolor* colourspace whitepoint.

RED_COLOR_WHITEPOINT : ndarray
"""

RED_COLOR_TO_XYZ_MATRIX = normalised_primary_matrix(
    RED_COLOR_PRIMARIES, RED_COLOR_WHITEPOINT)
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
    RED_COLOR_ILLUMINANT,
    RED_COLOR_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm)
"""
*REDcolor* colourspace.

RED_COLOR_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_2_PRIMARIES = np.array(
    [[0.878682510476129, 0.324964007409910],
     [0.300888714367432, 0.679054755790568],
     [0.095398694605615, -0.029379326834327]])
"""
*REDcolor2* colourspace primaries.

RED_COLOR_2_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_2_ILLUMINANT = RED_COLOR_ILLUMINANT
"""
*REDcolor2* colourspace whitepoint name as illuminant.

RED_COLOR_2_ILLUMINANT : unicode
"""

RED_COLOR_2_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*REDcolor2* colourspace whitepoint.

RED_COLOR_2_WHITEPOINT : ndarray
"""

RED_COLOR_2_TO_XYZ_MATRIX = normalised_primary_matrix(
    RED_COLOR_2_PRIMARIES, RED_COLOR_2_WHITEPOINT)
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
    RED_COLOR_2_ILLUMINANT,
    RED_COLOR_2_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_2_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm)
"""
*REDcolor2* colourspace.

RED_COLOR_2_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_3_PRIMARIES = np.array(
    [[0.701181035906413, 0.329014155583010],
     [0.300600304651563, 0.683788834268552],
     [0.108154455624011, -0.008688175786660]])
"""
*REDcolor3* colourspace primaries.

RED_COLOR_3_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_3_ILLUMINANT = RED_COLOR_ILLUMINANT
"""
*REDcolor3* colourspace whitepoint name as illuminant.

RED_COLOR_3_ILLUMINANT : unicode
"""

RED_COLOR_3_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*REDcolor3* colourspace whitepoint.

RED_COLOR_3_WHITEPOINT : ndarray
"""

RED_COLOR_3_TO_XYZ_MATRIX = normalised_primary_matrix(
    RED_COLOR_3_PRIMARIES, RED_COLOR_3_WHITEPOINT)
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
    RED_COLOR_3_ILLUMINANT,
    RED_COLOR_3_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_3_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm)
"""
*REDcolor3* colourspace.

RED_COLOR_3_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_4_PRIMARIES = np.array(
    [[0.701180591891983, 0.329013699115539],
     [0.300600395529389, 0.683788824257266],
     [0.145331946228869, 0.051616803622619]])
"""
*REDcolor4* colourspace primaries.

RED_COLOR_4_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_4_ILLUMINANT = RED_COLOR_ILLUMINANT
"""
*REDcolor4* colourspace whitepoint name as illuminant.

RED_COLOR_4_ILLUMINANT : unicode
"""

RED_COLOR_4_WHITEPOINT = RED_COLOR_WHITEPOINT
"""
*REDcolor4* colourspace whitepoint.

RED_COLOR_4_WHITEPOINT : ndarray
"""

RED_COLOR_4_TO_XYZ_MATRIX = normalised_primary_matrix(
    RED_COLOR_4_PRIMARIES, RED_COLOR_4_WHITEPOINT)
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
    RED_COLOR_4_ILLUMINANT,
    RED_COLOR_4_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_4_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm)
"""
*REDcolor4* colourspace.

RED_COLOR_4_COLOURSPACE : RGB_Colourspace
"""

DRAGON_COLOR_PRIMARIES = np.array(
    [[0.753044222784747, 0.327830576681599],
     [0.299570228480719, 0.700699321955751],
     [0.079642066734959, -0.054937951088786]])
"""
*DRAGONcolor* colourspace primaries.

DRAGON_COLOR_PRIMARIES : ndarray, (3, 2)
"""

DRAGON_COLOR_ILLUMINANT = RED_COLOR_ILLUMINANT
"""
*DRAGONcolor* colourspace whitepoint name as illuminant.

DRAGON_COLOR_ILLUMINANT : unicode
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
    DRAGON_COLOR_ILLUMINANT,
    DRAGON_COLOR_TO_XYZ_MATRIX,
    XYZ_TO_DRAGON_COLOR_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm)
"""
*DRAGONcolor* colourspace.

DRAGON_COLOR_COLOURSPACE : RGB_Colourspace
"""

DRAGON_COLOR_2_PRIMARIES = np.array(
    [[0.753044491143000, 0.327831029513214],
     [0.299570490451307, 0.700699415613996],
     [0.145011584277975, 0.051097125087887]])
"""
*DRAGONcolor2* colourspace primaries.

DRAGON_COLOR_2_PRIMARIES : ndarray, (3, 2)
"""

DRAGON_COLOR_2_ILLUMINANT = RED_COLOR_ILLUMINANT
"""
*DRAGONcolor2* colourspace whitepoint name as illuminant.

DRAGON_COLOR_2_ILLUMINANT : unicode
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
    DRAGON_COLOR_2_ILLUMINANT,
    DRAGON_COLOR_2_TO_XYZ_MATRIX,
    XYZ_TO_DRAGON_COLOR_2_MATRIX,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm)
"""
*DRAGONcolor2* colourspace.

DRAGON_COLOR_2_COLOURSPACE : RGB_Colourspace
"""

RED_WIDE_GAMUT_RGB_PRIMARIES = np.array(
    [[0.780308, 0.304253],
     [0.121595, 1.493994],
     [0.095612, -0.084589]])
"""
*REDWideGamutRGB* colourspace primaries.

RED_WIDE_GAMUT_RGB_PRIMARIES : ndarray, (3, 2)
"""

RED_WIDE_GAMUT_RGB_ILLUMINANT = 'D65'
"""
*REDWideGamutRGB* colourspace whitepoint name as illuminant.

RED_WIDE_GAMUT_RGB_ILLUMINANT : unicode
"""

RED_WIDE_GAMUT_RGB_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
        RED_WIDE_GAMUT_RGB_ILLUMINANT])
"""
*REDWideGamutRGB* colourspace whitepoint.

RED_WIDE_GAMUT_RGB_WHITEPOINT : ndarray
"""

RED_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = np.array(
    [[0.735275, 0.068609, 0.146571],
     [0.286694, 0.842979, -0.129673],
     [-0.079681, -0.347343, 1.516082]])
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
    RED_WIDE_GAMUT_RGB_ILLUMINANT,
    RED_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
    XYZ_TO_RED_WIDE_GAMUT_RGB_MATRIX,
    log_encoding_Log3G10,
    log_decoding_Log3G10)
"""
*REDWideGamutRGB* colourspace.

RED_WIDE_GAMUT_RGB_COLOURSPACE : RGB_Colourspace
"""
