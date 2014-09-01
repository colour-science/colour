#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ColorMatch RGB Colourspace
==========================

Defines the *ColorMatch RGB* colourspace:

-   :attr:`COLOR_MATCH_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  http://www.brucelindbloom.com/WorkingSpaceInfo.html
        (Last accessed 11 April 2014)
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['COLOR_MATCH_RGB_PRIMARIES',
           'COLOR_MATCH_RGB_WHITEPOINT',
           'COLOR_MATCH_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_COLOR_MATCH_RGB_MATRIX',
           'COLOR_MATCH_RGB_TRANSFER_FUNCTION',
           'COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION',
           'COLOR_MATCH_RGB_COLOURSPACE']

COLOR_MATCH_RGB_PRIMARIES = np.array(
    [[0.6300, 0.3400],
     [0.2950, 0.6050],
     [0.1500, 0.0750]])
"""
*ColorMatch RGB* colourspace primaries.

COLOR_MATCH_RGB_PRIMARIES : ndarray, (3, 2)
"""

COLOR_MATCH_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get('D50')
"""
*ColorMatch RGB* colourspace whitepoint.

COLOR_MATCH_RGB_WHITEPOINT : tuple
"""

COLOR_MATCH_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    COLOR_MATCH_RGB_PRIMARIES,
    COLOR_MATCH_RGB_WHITEPOINT)
"""
*ColorMatch RGB* colourspace to *CIE XYZ* colourspace matrix.

COLOR_MATCH_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_COLOR_MATCH_RGB_MATRIX = np.linalg.inv(COLOR_MATCH_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *ColorMatch RGB* colourspace matrix.

XYZ_TO_COLOR_MATCH_RGB_MATRIX : array_like, (3, 3)
"""

COLOR_MATCH_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 1.8)
"""
Transfer function from linear to *ColorMatch RGB* colourspace.

COLOR_MATCH_RGB_TRANSFER_FUNCTION : object
"""

COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 1.8
"""
Inverse transfer function from *ColorMatch RGB* colourspace to linear.

COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION : object
"""

COLOR_MATCH_RGB_COLOURSPACE = RGB_Colourspace(
    'ColorMatch RGB',
    COLOR_MATCH_RGB_PRIMARIES,
    COLOR_MATCH_RGB_WHITEPOINT,
    COLOR_MATCH_RGB_TO_XYZ_MATRIX,
    XYZ_TO_COLOR_MATCH_RGB_MATRIX,
    COLOR_MATCH_RGB_TRANSFER_FUNCTION,
    COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION)
"""
*ColorMatch RGB* colourspace.

COLOR_MATCH_RGB_COLOURSPACE : RGB_Colourspace
"""
