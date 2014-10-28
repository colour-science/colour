#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProPhoto RGB Colourspace
========================

Defines the *ProPhoto RGB* colourspace:

-   :attr:`PROPHOTO_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  ANSI. (2003). Specification of ROMM RGB. Retrieved from
        http://www.color.org/ROMMRGB.pdf
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

__all__ = ['PROPHOTO_RGB_PRIMARIES',
           'PROPHOTO_RGB_ILLUMINANT',
           'PROPHOTO_RGB_WHITEPOINT',
           'PROPHOTO_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_PROPHOTO_RGB_MATRIX',
           'PROPHOTO_RGB_TRANSFER_FUNCTION',
           'PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION',
           'PROPHOTO_RGB_COLOURSPACE']

PROPHOTO_RGB_PRIMARIES = np.array(
    [[0.7347, 0.2653],
     [0.1596, 0.8404],
     [0.0366, 0.0001]])
"""
*ProPhoto RGB* colourspace primaries.

PROPHOTO_RGB_PRIMARIES : ndarray, (3, 2)
"""

PROPHOTO_RGB_ILLUMINANT = 'D50'
"""
*ProPhoto RGB* colourspace whitepoint name as illuminant.

PROPHOTO_RGB_ILLUMINANT : unicode
"""

PROPHOTO_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(PROPHOTO_RGB_ILLUMINANT)
"""
*ProPhoto RGB* colourspace whitepoint.

PROPHOTO_RGB_WHITEPOINT : tuple
"""

PROPHOTO_RGB_TO_XYZ_MATRIX = np.array(
    [[7.97667235e-01, 1.35192231e-01, 3.13525290e-02],
     [2.88037454e-01, 7.11876883e-01, 8.56626476e-05],
     [0.00000000e+00, 0.00000000e+00, 8.25188285e-01]])
"""
*ProPhoto RGB* colourspace to *CIE XYZ* colourspace matrix.

PROPHOTO_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_PROPHOTO_RGB_MATRIX = np.linalg.inv(PROPHOTO_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *ProPhoto RGB* colourspace matrix.

XYZ_TO_PROPHOTO_RGB_MATRIX : array_like, (3, 3)
"""


def _prophoto_rgb_transfer_function(value):
    """
    Defines the *ProPhoto RGB* value colourspace transfer function.

    Parameters
    ----------
    value : numeric
        value.

    Returns
    -------
    numeric
        Companded value.
    """

    return value * 16 if value < 0.001953 else value ** (1 / 1.8)


def _prophoto_rgb_inverse_transfer_function(value):
    """
    Defines the *ProPhoto RGB* value colourspace inverse transfer
    function.

    Parameters
    ----------
    value : numeric
        value.

    Returns
    -------
    numeric
        Companded value.
    """

    return (value / 16
            if value < _prophoto_rgb_transfer_function(0.001953) else
            value ** 1.8)


PROPHOTO_RGB_TRANSFER_FUNCTION = _prophoto_rgb_transfer_function
"""
Transfer function from linear to *ProPhoto RGB* colourspace.

PROPHOTO_RGB_TRANSFER_FUNCTION : object
"""

PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION = (
    _prophoto_rgb_inverse_transfer_function)
"""
Inverse transfer function from *ProPhoto RGB* colourspace to linear.

PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION : object
"""

PROPHOTO_RGB_COLOURSPACE = RGB_Colourspace(
    'ProPhoto RGB',
    PROPHOTO_RGB_PRIMARIES,
    PROPHOTO_RGB_WHITEPOINT,
    PROPHOTO_RGB_ILLUMINANT,
    PROPHOTO_RGB_TO_XYZ_MATRIX,
    XYZ_TO_PROPHOTO_RGB_MATRIX,
    PROPHOTO_RGB_TRANSFER_FUNCTION,
    PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION)
"""
*ProPhoto RGB* colourspace.

PROPHOTO_RGB_COLOURSPACE : RGB_Colourspace
"""
