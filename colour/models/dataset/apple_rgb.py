#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apple RGB Colourspace
=====================

Defines the *Apple RGB* colourspace:

-   :attr:`APPLE_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  Lindbloom, B. (2014). RGB Working Space Information. Retrieved April
        11, 2014, from http://www.brucelindbloom.com/WorkingSpaceInfo.html
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

__all__ = ['APPLE_RGB_PRIMARIES',
           'APPLE_RGB_ILLUMINANT',
           'APPLE_RGB_WHITEPOINT',
           'APPLE_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_APPLE_RGB_MATRIX',
           'APPLE_RGB_TRANSFER_FUNCTION',
           'APPLE_RGB_INVERSE_TRANSFER_FUNCTION',
           'APPLE_RGB_COLOURSPACE']

APPLE_RGB_PRIMARIES = np.array(
    [[0.6250, 0.3400],
     [0.2800, 0.5950],
     [0.1550, 0.0700]])
"""
*Apple RGB* colourspace primaries.

APPLE_RGB_PRIMARIES : ndarray, (3, 2)
"""

APPLE_RGB_ILLUMINANT = 'D65'
"""
*Apple RGB* colourspace whitepoint name as illuminant.

APPLE_RGB_ILLUMINANT : unicode
"""

APPLE_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(APPLE_RGB_ILLUMINANT)
"""
*Apple RGB* colourspace whitepoint.

APPLE_RGB_WHITEPOINT : tuple
"""

APPLE_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(APPLE_RGB_PRIMARIES,
                                                    APPLE_RGB_WHITEPOINT)
"""
*Apple RGB* colourspace to *CIE XYZ* tristimulus values matrix.

APPLE_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_APPLE_RGB_MATRIX = np.linalg.inv(APPLE_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Apple RGB* colourspace matrix.

XYZ_TO_APPLE_RGB_MATRIX : array_like, (3, 3)
"""


def _apple_rgb_transfer_function(value):
    """
    Defines the *Apple RGB* colourspace transfer function.

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

    return value ** (1 / 1.8)


def _apple_rgb_inverse_transfer_function(value):
    """
    Defines the *Apple RGB* colourspace inverse transfer function.

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

    return value ** 1.8


APPLE_RGB_TRANSFER_FUNCTION = _apple_rgb_transfer_function
"""
Transfer function from linear to *Apple RGB* colourspace.

APPLE_RGB_TRANSFER_FUNCTION : object
"""

APPLE_RGB_INVERSE_TRANSFER_FUNCTION = _apple_rgb_inverse_transfer_function
"""
Inverse transfer function from *Apple RGB* colourspace to linear.

APPLE_RGB_INVERSE_TRANSFER_FUNCTION : object
"""

APPLE_RGB_COLOURSPACE = RGB_Colourspace(
    'Apple RGB',
    APPLE_RGB_PRIMARIES,
    APPLE_RGB_WHITEPOINT,
    APPLE_RGB_ILLUMINANT,
    APPLE_RGB_TO_XYZ_MATRIX,
    XYZ_TO_APPLE_RGB_MATRIX,
    APPLE_RGB_TRANSFER_FUNCTION,
    APPLE_RGB_INVERSE_TRANSFER_FUNCTION)
"""
*Apple RGB* colourspace.

APPLE_RGB_COLOURSPACE : RGB_Colourspace
"""
