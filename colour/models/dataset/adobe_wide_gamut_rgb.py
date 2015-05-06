#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adobe Wide Gamut RGB Colourspace
================================

Defines the *Adobe Wide Gamut RGB* colourspace:

-   :attr:`ADOBE_WIDE_GAMUT_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  Wikipedia. (n.d.). Wide-gamut RGB color space. Retrieved April 13,
        2014, from http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space
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

__all__ = ['ADOBE_WIDE_GAMUT_RGB_PRIMARIES',
           'ADOBE_WIDE_GAMUT_RGB_ILLUMINANT',
           'ADOBE_WIDE_GAMUT_RGB_WHITEPOINT',
           'ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX',
           'ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION',
           'ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION',
           'ADOBE_WIDE_GAMUT_RGB_COLOURSPACE']

ADOBE_WIDE_GAMUT_RGB_PRIMARIES = np.array(
    [[0.7347, 0.2653],
     [0.1152, 0.8264],
     [0.1566, 0.0177]])
"""
*Adobe Wide Gamut RGB* colourspace primaries.

ADOBE_WIDE_GAMUT_RGB_PRIMARIES : ndarray, (3, 2)
"""

ADOBE_WIDE_GAMUT_RGB_ILLUMINANT = 'D50'
"""
*Adobe Wide Gamut RGB* colourspace whitepoint name as illuminant.

ADOBE_WIDE_GAMUT_RGB_ILLUMINANT : unicode
"""

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(
    ADOBE_WIDE_GAMUT_RGB_ILLUMINANT)
"""
*Adobe Wide Gamut RGB* colourspace whitepoint.

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT : tuple
"""

ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
    ADOBE_WIDE_GAMUT_RGB_WHITEPOINT)
"""
*Adobe Wide Gamut RGB* colourspace to *CIE XYZ* tristimulus values matrix.

ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX = np.linalg.inv(
    ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Adobe Wide Gamut RGB* colourspace matrix.

XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX : array_like, (3, 3)
"""


def _adobe_wide_gamut_rgb_transfer_function(value):
    """
    Defines the *Adobe Wide Gamut RGB* colourspace transfer function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return value ** (1 / (563 / 256))


def _adobe_wide_gamut_rgb_inverse_transfer_function(value):
    """
    Defines the *Adobe Wide Gamut RGB* colourspace inverse transfer function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return value ** (563 / 256)


ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION = (
    _adobe_wide_gamut_rgb_transfer_function)
"""
Transfer function from linear to *Adobe Wide Gamut RGB* colourspace.

ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION : object
"""

ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION = (
    _adobe_wide_gamut_rgb_inverse_transfer_function)
"""
Inverse transfer function from *Adobe Wide Gamut RGB* colourspace to linear.

ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION : object
"""

ADOBE_WIDE_GAMUT_RGB_COLOURSPACE = RGB_Colourspace(
    'Adobe Wide Gamut RGB',
    ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
    ADOBE_WIDE_GAMUT_RGB_WHITEPOINT,
    ADOBE_WIDE_GAMUT_RGB_ILLUMINANT,
    ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX,
    ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION,
    ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION)
"""
*Adobe Wide Gamut RGB* colourspace.

ADOBE_WIDE_GAMUT_RGB_COLOURSPACE : RGB_Colourspace
"""
