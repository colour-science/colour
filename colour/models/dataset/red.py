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

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Mansencal, T. (2015). RED Colourspaces Derivation. Retrieved May 20,
        2015, from http://colour-science.org/posts/red-colourspaces-derivation
.. [2]  Sony Imageworks. (2012). make.py. Retrieved November 27, 2014, from
        https://github.com/imageworks/OpenColorIO-Configs/\
blob/master/nuke-default/make.py
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RED_COLOR_PRIMARIES',
           'RED_COLOR_ILLUMINANT',
           'RED_COLOR_WHITEPOINT',
           'RED_COLOR_TO_XYZ_MATRIX',
           'XYZ_TO_RED_COLOR_MATRIX',
           'RED_LOG_FILM_OECF',
           'RED_LOG_FILM_EOCF',
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
           'DRAGON_COLOR_2_COLOURSPACE']

RED_COLOR_PRIMARIES = np.array(
    [[0.6997470012907312, 0.3290469303126368],
     [0.3042640390235472, 0.6236411451291149],
     [0.1349139612964870, 0.0347174412813451]])
"""
*REDcolor* colourspace primaries.

RED_COLOR_PRIMARIES : ndarray, (3, 2)
"""

RED_COLOR_ILLUMINANT = 'D60'
"""
*REDcolor* colourspace whitepoint name as illuminant.

RED_COLOR_ILLUMINANT : unicode
"""

RED_COLOR_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(RED_COLOR_ILLUMINANT)
"""
*REDcolor* colourspace whitepoint.

RED_COLOR_WHITEPOINT : tuple
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


def _linear_to_red_log_film(
        value,
        black_offset=10 ** ((0 - 1023) / 511)):
    """
    Defines the *REDLogFilm* opto-electronic conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return ((1023 +
             511 * np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def _red_log_film_to_linear(
        value,
        black_offset=10 ** ((0 - 1023) / 511)):
    """
    Defines the *REDLogFilm* electro-optical conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return (((10 **
              ((1023 * value - 1023) / 511)) - black_offset) /
            (1 - black_offset))


RED_LOG_FILM_OECF = _linear_to_red_log_film
"""
Opto-electronic conversion function of *REDLogFilm*.

RED_LOG_FILM_OECF : object
"""

RED_LOG_FILM_EOCF = _red_log_film_to_linear
"""
Electro-optical conversion function of *REDLogFilm* to linear.

RED_LOG_FILM_EOCF : object
"""

RED_COLOR_COLOURSPACE = RGB_Colourspace(
    'REDcolor',
    RED_COLOR_PRIMARIES,
    RED_COLOR_WHITEPOINT,
    RED_COLOR_ILLUMINANT,
    RED_COLOR_TO_XYZ_MATRIX,
    XYZ_TO_RED_COLOR_MATRIX,
    RED_LOG_FILM_OECF,
    RED_LOG_FILM_EOCF)
"""
*REDcolor* colourspace.

RED_COLOR_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_2_PRIMARIES = np.array(
    [[0.8786825104761286, 0.3249640074099105],
     [0.3008887143674324, 0.6790547557905675],
     [0.0953986946056151, -0.0293793268343266]])
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

RED_COLOR_2_WHITEPOINT : tuple
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
    RED_LOG_FILM_OECF,
    RED_LOG_FILM_EOCF)
"""
*REDcolor2* colourspace.

RED_COLOR_2_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_3_PRIMARIES = np.array(
    [[0.7011810359064131, 0.3290141555830101],
     [0.3006003046515633, 0.6837888342685519],
     [0.1081544556240110, -0.0086881757866604]])
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

RED_COLOR_3_WHITEPOINT : tuple
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
    RED_LOG_FILM_OECF,
    RED_LOG_FILM_EOCF)
"""
*REDcolor3* colourspace.

RED_COLOR_3_COLOURSPACE : RGB_Colourspace
"""

RED_COLOR_4_PRIMARIES = np.array(
    [[0.7011805918919830, 0.3290136991155385],
     [0.3006003955293892, 0.6837888242572663],
     [0.1453319462288687, 0.0516168036226188]])
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

RED_COLOR_4_WHITEPOINT : tuple
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
    RED_LOG_FILM_OECF,
    RED_LOG_FILM_EOCF)
"""
*REDcolor4* colourspace.

RED_COLOR_4_COLOURSPACE : RGB_Colourspace
"""

DRAGON_COLOR_PRIMARIES = np.array(
    [[0.7530442227847470, 0.3278305766815993],
     [0.2995702284807185, 0.7006993219557512],
     [0.0796420667349588, -0.0549379510887859]])
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

DRAGON_COLOR_WHITEPOINT : tuple
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
    RED_LOG_FILM_OECF,
    RED_LOG_FILM_EOCF)
"""
*DRAGONcolor* colourspace.

DRAGON_COLOR_COLOURSPACE : RGB_Colourspace
"""

DRAGON_COLOR_2_PRIMARIES = np.array(
    [[0.7530444911429997, 0.3278310295132136],
     [0.2995704904513070, 0.7006994156139956],
     [0.1450115842779754, 0.0510971250878873]])
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

DRAGON_COLOR_2_WHITEPOINT : tuple
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
    RED_LOG_FILM_OECF,
    RED_LOG_FILM_EOCF)
"""
*DRAGONcolor2* colourspace.

DRAGON_COLOR_2_COLOURSPACE : RGB_Colourspace
"""
