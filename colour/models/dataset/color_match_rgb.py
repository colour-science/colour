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
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

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

__all__ = ['COLOR_MATCH_RGB_PRIMARIES',
           'COLOR_MATCH_RGB_ILLUMINANT',
           'COLOR_MATCH_RGB_WHITEPOINT',
           'COLOR_MATCH_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_COLOR_MATCH_RGB_MATRIX',
           'COLOR_MATCH_RGB_OECF',
           'COLOR_MATCH_RGB_EOCF',
           'COLOR_MATCH_RGB_COLOURSPACE']

COLOR_MATCH_RGB_PRIMARIES = np.array(
    [[0.6300, 0.3400],
     [0.2950, 0.6050],
     [0.1500, 0.0750]])
"""
*ColorMatch RGB* colourspace primaries.

COLOR_MATCH_RGB_PRIMARIES : ndarray, (3, 2)
"""

COLOR_MATCH_RGB_ILLUMINANT = 'D50'
"""
*ColorMatch RGB* colourspace whitepoint name as illuminant.

COLOR_MATCH_RGB_ILLUMINANT : unicode
"""

COLOR_MATCH_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(COLOR_MATCH_RGB_ILLUMINANT)
"""
*ColorMatch RGB* colourspace whitepoint.

COLOR_MATCH_RGB_WHITEPOINT : tuple
"""

COLOR_MATCH_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    COLOR_MATCH_RGB_PRIMARIES,
    COLOR_MATCH_RGB_WHITEPOINT)
"""
*ColorMatch RGB* colourspace to *CIE XYZ* tristimulus values matrix.

COLOR_MATCH_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_COLOR_MATCH_RGB_MATRIX = np.linalg.inv(COLOR_MATCH_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *ColorMatch RGB* colourspace matrix.

XYZ_TO_COLOR_MATCH_RGB_MATRIX : array_like, (3, 3)
"""


def _color_match_rgb_OECF(value):
    """
    Defines the *ColorMatch RGB* colourspace opto-electronic conversion
    function.

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


def _color_match_rgb_EOCF(value):
    """
    Defines the *ColorMatch RGB* colourspace electro-optical conversion
    function.

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


COLOR_MATCH_RGB_OECF = _color_match_rgb_OECF
"""
Opto-electronic conversion function of *ColorMatch RGB*
colourspace.

COLOR_MATCH_RGB_OECF : object
"""

COLOR_MATCH_RGB_EOCF = (
    _color_match_rgb_EOCF)
"""
Electro-optical conversion function of *ColorMatch RGB* colourspace to
linear.

COLOR_MATCH_RGB_EOCF : object
"""

COLOR_MATCH_RGB_COLOURSPACE = RGB_Colourspace(
    'ColorMatch RGB',
    COLOR_MATCH_RGB_PRIMARIES,
    COLOR_MATCH_RGB_WHITEPOINT,
    COLOR_MATCH_RGB_ILLUMINANT,
    COLOR_MATCH_RGB_TO_XYZ_MATRIX,
    XYZ_TO_COLOR_MATCH_RGB_MATRIX,
    COLOR_MATCH_RGB_OECF,
    COLOR_MATCH_RGB_EOCF)
"""
*ColorMatch RGB* colourspace.

COLOR_MATCH_RGB_COLOURSPACE : RGB_Colourspace
"""
