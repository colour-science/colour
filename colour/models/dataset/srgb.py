#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sRGB Colourspace
================

Defines the *sRGB* colourspace:

-   :attr:`sRGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  `Recommendation ITU-R BT.709-5 - Parameter values for the HDTV
        standards for production and international programme exchange
        <http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf>`_  # noqa
        (Last accessed 24 February 2014)
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

__all__ = ['sRGB_PRIMARIES',
           'sRGB_WHITEPOINT',
           'sRGB_TO_XYZ_MATRIX',
           'XYZ_TO_sRGB_MATRIX',
           'sRGB_TRANSFER_FUNCTION',
           'sRGB_INVERSE_TRANSFER_FUNCTION',
           'sRGB_COLOURSPACE']

sRGB_PRIMARIES = np.array(
    [[0.6400, 0.3300],
     [0.3000, 0.6000],
     [0.1500, 0.0600]])
"""
*sRGB* colourspace primaries.

sRGB_PRIMARIES : ndarray, (3, 2)
"""

sRGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get('D65')
"""
*sRGB* colourspace whitepoint.

sRGB_WHITEPOINT : tuple
"""

sRGB_TO_XYZ_MATRIX = np.array(
    [[0.41238656, 0.35759149, 0.18045049],
     [0.21263682, 0.71518298, 0.0721802],
     [0.01933062, 0.11919716, 0.95037259]])
"""
*sRGB* colourspace to *CIE XYZ* colourspace matrix.

sRGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_sRGB_MATRIX = np.linalg.inv(sRGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *sRGB* colourspace matrix.

XYZ_TO_sRGB_MATRIX : array_like, (3, 3)
"""

sRGB_TRANSFER_FUNCTION = lambda x: (
    x * 12.92 if x <= 0.0031308 else 1.055 * (x ** (1 / 2.4)) - 0.055)
"""
Transfer function from linear to *sRGB* colourspace.

sRGB_TRANSFER_FUNCTION : object
"""

sRGB_INVERSE_TRANSFER_FUNCTION = lambda x: (
    x / 12.92 if x <= 0.0031308 else ((x + 0.055) / 1.055) ** 2.4)
"""
Inverse transfer function from *sRGB* colourspace to linear.

sRGB_INVERSE_TRANSFER_FUNCTION : object
"""

sRGB_COLOURSPACE = RGB_Colourspace(
    'sRGB',
    sRGB_PRIMARIES,
    sRGB_WHITEPOINT,
    sRGB_TO_XYZ_MATRIX,
    XYZ_TO_sRGB_MATRIX,
    sRGB_TRANSFER_FUNCTION,
    sRGB_INVERSE_TRANSFER_FUNCTION)
"""
*sRGB* colourspace.

sRGB_COLOURSPACE : RGB_Colourspace
"""
