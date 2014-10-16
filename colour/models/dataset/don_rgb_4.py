#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Don RGB 4 Colourspace
=====================

Defines the *Don RGB 4* colourspace:

-   :attr:`DON_RGB_4_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  HutchColor. (n.d.). DonRGB4 (4 K). Retrieved from
        http://www.hutchcolor.com/profiles/DonRGB4.zip
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

__all__ = ['DON_RGB_4_PRIMARIES',
           'DON_RGB_4_ILLUMINANT',
           'DON_RGB_4_WHITEPOINT',
           'DON_RGB_4_TO_XYZ_MATRIX',
           'XYZ_TO_DON_RGB_4_MATRIX',
           'DON_RGB_4_TRANSFER_FUNCTION',
           'DON_RGB_4_INVERSE_TRANSFER_FUNCTION',
           'DON_RGB_4_COLOURSPACE']

DON_RGB_4_PRIMARIES = np.array(
    [[0.69612068965517238, 0.29956896551724138],
     [0.21468298109010012, 0.7652947719688542],
     [0.12993762993762992, 0.035343035343035345]])
"""
*Don RGB 4* colourspace primaries.

DON_RGB_4_PRIMARIES : ndarray, (3, 2)
"""

DON_RGB_4_ILLUMINANT = 'D50'
"""
*Don RGB 4* colourspace whitepoint name as illuminant.

DON_RGB_4_ILLUMINANT : unicode
"""

DON_RGB_4_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(DON_RGB_4_ILLUMINANT)
"""
*Don RGB 4* colourspace whitepoint.

DON_RGB_4_WHITEPOINT : tuple
"""

DON_RGB_4_TO_XYZ_MATRIX = normalised_primary_matrix(DON_RGB_4_PRIMARIES,
                                                    DON_RGB_4_WHITEPOINT)
"""
*Don RGB 4* colourspace to *CIE XYZ* colourspace matrix.

DON_RGB_4_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DON_RGB_4_MATRIX = np.linalg.inv(DON_RGB_4_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *Don RGB 4* colourspace matrix.

XYZ_TO_DON_RGB_4_MATRIX : array_like, (3, 3)
"""


def _don_rgb_4_transfer_function(value):
    """
    Defines the *Don RGB 4* value colourspace transfer function.

    Parameters
    ----------
    value : numeric
        value.

    Returns
    -------
    numeric
        Companded value.
    """

    return value ** (1 / 2.2)


def _don_rgb_4_inverse_transfer_function(value):
    """
    Defines the *Don RGB 4* value colourspace inverse transfer
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

    return value ** 2.2


DON_RGB_4_TRANSFER_FUNCTION = _don_rgb_4_transfer_function
"""
Transfer function from linear to *Don RGB 4* colourspace.

DON_RGB_4_TRANSFER_FUNCTION : object
"""

DON_RGB_4_INVERSE_TRANSFER_FUNCTION = _don_rgb_4_inverse_transfer_function
"""
Inverse transfer function from *Don RGB 4* colourspace to linear.

DON_RGB_4_INVERSE_TRANSFER_FUNCTION : object
"""

DON_RGB_4_COLOURSPACE = RGB_Colourspace(
    'Don RGB 4',
    DON_RGB_4_PRIMARIES,
    DON_RGB_4_WHITEPOINT,
    DON_RGB_4_ILLUMINANT,
    DON_RGB_4_TO_XYZ_MATRIX,
    XYZ_TO_DON_RGB_4_MATRIX,
    DON_RGB_4_TRANSFER_FUNCTION,
    DON_RGB_4_INVERSE_TRANSFER_FUNCTION)
"""
*Don RGB 4* colourspace.

DON_RGB_4_COLOURSPACE : RGB_Colourspace
"""
