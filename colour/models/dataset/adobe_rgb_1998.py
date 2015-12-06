#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adobe RGB 1998 Colourspace
==========================

Defines the *Adobe RGB 1998* colourspace:

-   :attr:`ADOBE_RGB_1998_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Adobe Systems. (2005). Adobe RGB (1998) Color Image Encoding.
        Retrieved from http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ADOBE_RGB_1998_PRIMARIES',
           'ADOBE_RGB_1998_ILLUMINANT',
           'ADOBE_RGB_1998_WHITEPOINT',
           'ADOBE_RGB_1998_TO_XYZ_MATRIX',
           'XYZ_TO_ADOBE_RGB_1998_MATRIX',
           'ADOBE_RGB_1998_OECF',
           'ADOBE_RGB_1998_EOCF',
           'ADOBE_RGB_1998_COLOURSPACE']

ADOBE_RGB_1998_PRIMARIES = np.array(
    [[0.6400, 0.3300],
     [0.2100, 0.7100],
     [0.1500, 0.0600]])
"""
*Adobe RGB 1998* colourspace primaries.

ADOBE_RGB_1998_PRIMARIES : ndarray, (3, 2)
"""

ADOBE_RGB_1998_ILLUMINANT = 'D65'
"""
*Adobe RGB 1998* colourspace whitepoint name as illuminant.

ADOBE_RGB_1998_ILLUMINANT : unicode
"""

ADOBE_RGB_1998_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(ADOBE_RGB_1998_ILLUMINANT)
"""
*Adobe RGB 1998* colourspace whitepoint.

ADOBE_RGB_1998_WHITEPOINT : tuple
"""

ADOBE_RGB_1998_TO_XYZ_MATRIX = np.array(
    [[0.57666809, 0.18556195, 0.1881985],
     [0.29734449, 0.62737611, 0.0752794],
     [0.02703132, 0.07069027, 0.99117879]])
"""
*Adobe RGB 1998* colourspace to *CIE XYZ* tristimulus values matrix.

ADOBE_RGB_1998_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ADOBE_RGB_1998_MATRIX = np.linalg.inv(ADOBE_RGB_1998_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Adobe RGB 1998* colourspace matrix.

XYZ_TO_ADOBE_RGB_1998_MATRIX : array_like, (3, 3)
"""


def _adobe_rgb_1998_OECF(value):
    """
    Defines the *Adobe RGB 1998* colourspace opto-electronic conversion
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

    return value ** (1 / (563 / 256))


def _adobe_rgb_1998_EOCF(value):
    """
    Defines the *Adobe RGB 1998* colourspace electro-optical conversion
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

    return value ** (563 / 256)


ADOBE_RGB_1998_OECF = _adobe_rgb_1998_OECF
"""
Opto-electronic conversion function of *Adobe RGB 1998*
colourspace.

ADOBE_RGB_1998_OECF : object
"""

ADOBE_RGB_1998_EOCF = (
    _adobe_rgb_1998_EOCF)
"""
Electro-optical conversion function of *Adobe RGB 1998* colourspace to
linear.

ADOBE_RGB_1998_EOCF : object
"""

ADOBE_RGB_1998_COLOURSPACE = RGB_Colourspace(
    'Adobe RGB 1998',
    ADOBE_RGB_1998_PRIMARIES,
    ADOBE_RGB_1998_WHITEPOINT,
    ADOBE_RGB_1998_ILLUMINANT,
    ADOBE_RGB_1998_TO_XYZ_MATRIX,
    XYZ_TO_ADOBE_RGB_1998_MATRIX,
    ADOBE_RGB_1998_OECF,
    ADOBE_RGB_1998_EOCF)
"""
*Adobe RGB 1998* colourspace.

ADOBE_RGB_1998_COLOURSPACE : RGB_Colourspace
"""
