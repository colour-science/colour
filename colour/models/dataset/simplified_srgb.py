#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified sRGB Colourspace
===========================

Defines the *Simplified sRGB* colourspace:

-   :attr:`SIMPLIFIED_sRGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['SIMPLIFIED_sRGB_PRIMARIES',
           'SIMPLIFIED_sRGB_ILLUMINANT',
           'SIMPLIFIED_sRGB_WHITEPOINT',
           'SIMPLIFIED_sRGB_TO_XYZ_MATRIX',
           'SIMPLIFIED_XYZ_TO_sRGB_MATRIX',
           'SIMPLIFIED_sRGB_TRANSFER_FUNCTION',
           'SIMPLIFIED_sRGB_INVERSE_TRANSFER_FUNCTION',
           'SIMPLIFIED_sRGB_COLOURSPACE']

SIMPLIFIED_sRGB_PRIMARIES = np.array(
    [[0.6400, 0.3300],
     [0.3000, 0.6000],
     [0.1500, 0.0600]])
"""
*Simplified sRGB* colourspace primaries.

SIMPLIFIED_sRGB_PRIMARIES : ndarray, (3, 2)
"""

SIMPLIFIED_sRGB_ILLUMINANT = 'D65'
"""
*Simplified sRGB* colourspace whitepoint name as illuminant.

SIMPLIFIED_sRGB_WHITEPOINT : unicode
"""

SIMPLIFIED_sRGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(SIMPLIFIED_sRGB_ILLUMINANT)
"""
*Simplified sRGB* colourspace whitepoint.

SIMPLIFIED_sRGB_WHITEPOINT : tuple
"""

SIMPLIFIED_sRGB_TO_XYZ_MATRIX = np.array(
    [[0.41238656, 0.35759149, 0.18045049],
     [0.21263682, 0.71518298, 0.0721802],
     [0.01933062, 0.11919716, 0.95037259]])
"""
*Simplified sRGB* colourspace to *CIE XYZ* colourspace matrix.

SIMPLIFIED_sRGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_SIMPLIFIED_sRGB_MATRIX = np.linalg.inv(SIMPLIFIED_sRGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *Simplified sRGB* colourspace matrix.

XYZ_TO_SIMPLIFIED_sRGB_MATRIX : array_like, (3, 3)
"""


def _simplified_srgb_transfer_function(value):
    """
    Defines the *Simplified sRGB* colourspace transfer function.

    Parameters
    ----------
    value : numeric
        Value.

    Returns
    -------
    numeric
        Companded value.
    """

    return value ** (1 / 2.2)


def _simplified_srgb_inverse_transfer_function(value):
    """
    Defines the *Simplified sRGB* colourspace inverse transfer
    function.

    Parameters
    ----------
    value : numeric
        Value.

    Returns
    -------
    numeric
        Companded value.
    """

    return value ** 2.2


SIMPLIFIED_sRGB_TRANSFER_FUNCTION = _simplified_srgb_transfer_function
"""
Transfer function from linear to *Simplified sRGB* colourspace.

SIMPLIFIED_sRGB_TRANSFER_FUNCTION : object
"""

SIMPLIFIED_sRGB_INVERSE_TRANSFER_FUNCTION = _simplified_srgb_inverse_transfer_function
"""
Inverse transfer function from *Simplified sRGB* colourspace to linear.

SIMPLIFIED_sRGB_INVERSE_TRANSFER_FUNCTION : object
"""

SIMPLIFIED_sRGB_COLOURSPACE = RGB_Colourspace(
    'Simplified sRGB',
    SIMPLIFIED_sRGB_PRIMARIES,
    SIMPLIFIED_sRGB_WHITEPOINT,
    SIMPLIFIED_sRGB_ILLUMINANT,
    SIMPLIFIED_sRGB_TO_XYZ_MATRIX,
    XYZ_TO_SIMPLIFIED_sRGB_MATRIX,
    SIMPLIFIED_sRGB_TRANSFER_FUNCTION,
    SIMPLIFIED_sRGB_INVERSE_TRANSFER_FUNCTION)
"""
*Simplified sRGB* colourspace.

SIMPLIFIED_sRGB_COLOURSPACE : RGB_Colourspace
"""
