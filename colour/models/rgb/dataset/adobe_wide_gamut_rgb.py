# -*- coding: utf-8 -*-
"""
Adobe Wide Gamut RGB Colourspace
================================

Defines the *Adobe Wide Gamut RGB* colourspace:

-   :attr:`colour.models.ADOBE_WIDE_GAMUT_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Wikipedia2004c` : Wikipedia. (2004). Wide-gamut RGB color space.
    Retrieved April 13, 2014, from
    http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, gamma_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ADOBE_WIDE_GAMUT_RGB_PRIMARIES', 'ADOBE_WIDE_GAMUT_RGB_WHITEPOINT_NAME',
    'ADOBE_WIDE_GAMUT_RGB_WHITEPOINT', 'ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX',
    'XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX', 'ADOBE_WIDE_GAMUT_RGB_COLOURSPACE'
]

ADOBE_WIDE_GAMUT_RGB_PRIMARIES = np.array([
    [0.7347, 0.2653],
    [0.1152, 0.8264],
    [0.1566, 0.0177],
])
"""
*Adobe Wide Gamut RGB* colourspace primaries.

ADOBE_WIDE_GAMUT_RGB_PRIMARIES : ndarray, (3, 2)
"""

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT_NAME = 'D50'
"""
*Adobe Wide Gamut RGB* colourspace whitepoint name.

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT_NAME : unicode
"""

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
        ADOBE_WIDE_GAMUT_RGB_WHITEPOINT_NAME])
"""
*Adobe Wide Gamut RGB* colourspace whitepoint.

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT : ndarray
"""

ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    ADOBE_WIDE_GAMUT_RGB_PRIMARIES, ADOBE_WIDE_GAMUT_RGB_WHITEPOINT)
"""
*Adobe Wide Gamut RGB* colourspace to *CIE XYZ* tristimulus values matrix.

ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX = np.linalg.inv(
    ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Adobe Wide Gamut RGB* colourspace matrix.

XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX : array_like, (3, 3)
"""

ADOBE_WIDE_GAMUT_RGB_COLOURSPACE = RGB_Colourspace(
    'Adobe Wide Gamut RGB',
    ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
    ADOBE_WIDE_GAMUT_RGB_WHITEPOINT,
    ADOBE_WIDE_GAMUT_RGB_WHITEPOINT_NAME,
    ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX,
    partial(gamma_function, exponent=1 / (563 / 256)),
    partial(gamma_function, exponent=563 / 256),
)
ADOBE_WIDE_GAMUT_RGB_COLOURSPACE.__doc__ = """
*Adobe Wide Gamut RGB* colourspace.

References
----------
:cite:`Wikipedia2004c`

ADOBE_WIDE_GAMUT_RGB_COLOURSPACE : RGB_Colourspace
"""
