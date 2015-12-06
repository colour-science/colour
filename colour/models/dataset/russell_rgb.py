#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Russell RGB Colourspace
=======================

Defines the *Russell RGB* colourspace:

-   :attr:`RUSSELL_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Cottrell, R. (n.d.). The Russell RGB working color space. Retrieved
        from http://www.russellcottrell.com/photo/downloads/RussellRGB.icc
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry.dataset import ILLUMINANTS
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RUSSELL_RGB_PRIMARIES',
           'RUSSELL_RGB_ILLUMINANT',
           'RUSSELL_RGB_WHITEPOINT',
           'RUSSELL_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_RUSSELL_RGB_MATRIX',
           'RUSSELL_RGB_OECF',
           'RUSSELL_RGB_EOCF',
           'RUSSELL_RGB_COLOURSPACE']

RUSSELL_RGB_PRIMARIES = np.array(
    [[0.6900, 0.3100],
     [0.1800, 0.7700],
     [0.1000, 0.0200]])
"""
*Russell RGB* colourspace primaries.

RUSSELL_RGB_PRIMARIES : ndarray, (3, 2)
"""

RUSSELL_RGB_ILLUMINANT = 'D55'
"""
*Russell RGB* colourspace whitepoint name as illuminant.

RUSSELL_RGB_ILLUMINANT : unicode
"""

RUSSELL_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(RUSSELL_RGB_ILLUMINANT)
"""
*Russell RGB* colourspace whitepoint.

RUSSELL_RGB_WHITEPOINT : tuple
"""

RUSSELL_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    RUSSELL_RGB_PRIMARIES, RUSSELL_RGB_WHITEPOINT)
"""
*Russell RGB* colourspace to *CIE XYZ* tristimulus values matrix.

RUSSELL_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_RUSSELL_RGB_MATRIX = np.linalg.inv(RUSSELL_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Russell RGB* colourspace matrix.

XYZ_TO_RUSSELL_RGB_MATRIX : array_like, (3, 3)
"""


def _russell_rgb_OECF(value):
    """
    Defines the *Russell RGB* colourspace opto-electronic conversion function.

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

    return value ** (1 / 2.2)


def _russell_rgb_EOCF(value):
    """
    Defines the *Russell RGB* colourspace electro-optical conversion function.

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

    return value ** 2.2


RUSSELL_RGB_OECF = _russell_rgb_OECF
"""
Opto-electronic conversion function of *Russell RGB* colourspace.

RUSSELL_RGB_OECF : object
"""

RUSSELL_RGB_EOCF = _russell_rgb_EOCF
"""
Electro-optical conversion function of *Russell RGB* colourspace.

RUSSELL_RGB_EOCF : object
"""

RUSSELL_RGB_COLOURSPACE = RGB_Colourspace(
    'Russell RGB',
    RUSSELL_RGB_PRIMARIES,
    RUSSELL_RGB_WHITEPOINT,
    RUSSELL_RGB_ILLUMINANT,
    RUSSELL_RGB_TO_XYZ_MATRIX,
    XYZ_TO_RUSSELL_RGB_MATRIX,
    RUSSELL_RGB_OECF,
    RUSSELL_RGB_EOCF)
"""
*Russell RGB* colourspace.

RUSSELL_RGB_COLOURSPACE : RGB_Colourspace
"""
