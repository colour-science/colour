#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE RGB Colourspace
===================

Defines the *CIE RGB* colourspace:

-   :attr:`CIE_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  http://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright.E2.80.93Guild_data  # noqa
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

__all__ = ['CIE_RGB_PRIMARIES',
           'CIE_RGB_WHITEPOINT',
           'CIE_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_CIE_RGB_MATRIX',
           'CIE_RGB_TRANSFER_FUNCTION',
           'CIE_RGB_INVERSE_TRANSFER_FUNCTION',
           'CIE_RGB_COLOURSPACE']

CIE_RGB_PRIMARIES = np.array(
    [[0.7350, 0.2650],
     [0.2740, 0.7170],
     [0.1670, 0.0090]])
"""
*CIE RGB* colourspace primaries.

CIE_RGB_PRIMARIES : ndarray, (3, 2)
"""

CIE_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get('E')
"""
*CIE RGB* colourspace whitepoint.

CIE_RGB_WHITEPOINT : tuple
"""

CIE_RGB_TO_XYZ_MATRIX = (1 / 0.17697 *
                         np.array([[0.49, 0.31, 0.20],
                                   [0.17697, 0.81240, 0.01063],
                                   [0.00, 0.01, 0.99]]))
"""
*CIE RGB* colourspace to *CIE XYZ* colourspace matrix.

CIE_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_CIE_RGB_MATRIX = np.linalg.inv(CIE_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *CIE RGB* colourspace matrix.

XYZ_TO_CIE_RGB_MATRIX : array_like, (3, 3)
"""

CIE_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)
"""
Transfer function from linear to *CIE RGB* colourspace.

CIE_RGB_TRANSFER_FUNCTION : object
"""

CIE_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2
"""
Inverse transfer function from *CIE RGB* colourspace to linear.

CIE_RGB_INVERSE_TRANSFER_FUNCTION : object
"""

CIE_RGB_COLOURSPACE = RGB_Colourspace(
    'CIE RGB',
    CIE_RGB_PRIMARIES,
    CIE_RGB_WHITEPOINT,
    CIE_RGB_TO_XYZ_MATRIX,
    XYZ_TO_CIE_RGB_MATRIX,
    CIE_RGB_TRANSFER_FUNCTION,
    CIE_RGB_INVERSE_TRANSFER_FUNCTION)
"""
*CIE RGB* colourspace.

CIE_RGB_COLOURSPACE : RGB_Colourspace
"""
