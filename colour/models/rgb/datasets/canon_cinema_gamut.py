# -*- coding: utf-8 -*-
"""
Canon Cinema Gamut Colourspace
==============================

Defines the *Canon Cinema Gamut* colourspace:

-   :attr:`colour.models.CINEMA_GAMUT_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Canon2014a` : Canon. (2014). EOS C500 Firmware Update. Retrieved
    August 27, 2016, from https://www.usa.canon.com/internet/portal/us/home/\
explore/product-showcases/cameras-and-lenses/cinema-eos-firmware/c500
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, linear_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CINEMA_GAMUT_PRIMARIES', 'CINEMA_GAMUT_WHITEPOINT_NAME',
    'CINEMA_GAMUT_WHITEPOINT', 'CINEMA_GAMUT_TO_XYZ_MATRIX',
    'XYZ_TO_CINEMA_GAMUT_MATRIX', 'CINEMA_GAMUT_COLOURSPACE'
]

CINEMA_GAMUT_PRIMARIES = np.array([
    [0.7400, 0.2700],
    [0.1700, 1.1400],
    [0.0800, -0.1000],
])
"""
*Canon Cinema Gamut* colourspace primaries.

CINEMA_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

CINEMA_GAMUT_WHITEPOINT_NAME = 'D65'
"""
*Canon Cinema Gamut* colourspace whitepoint name.

CINEMA_GAMUT_WHITEPOINT_NAME : unicode
"""

CINEMA_GAMUT_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    CINEMA_GAMUT_WHITEPOINT_NAME])
"""
*Canon Cinema Gamut* colourspace whitepoint.

CINEMA_GAMUT_WHITEPOINT : ndarray
"""

CINEMA_GAMUT_TO_XYZ_MATRIX = normalised_primary_matrix(
    CINEMA_GAMUT_PRIMARIES, CINEMA_GAMUT_WHITEPOINT)
"""
*Canon Cinema Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

CINEMA_GAMUT_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_CINEMA_GAMUT_MATRIX = np.linalg.inv(CINEMA_GAMUT_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Canon Cinema Gamut* colourspace matrix.

XYZ_TO_CINEMA_GAMUT_MATRIX : array_like, (3, 3)
"""

CINEMA_GAMUT_COLOURSPACE = RGB_Colourspace(
    'Cinema Gamut',
    CINEMA_GAMUT_PRIMARIES,
    CINEMA_GAMUT_WHITEPOINT,
    CINEMA_GAMUT_WHITEPOINT_NAME,
    CINEMA_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_CINEMA_GAMUT_MATRIX,
    linear_function,
    linear_function,
)
CINEMA_GAMUT_COLOURSPACE.__doc__ = """
*Canon Cinema Gamut* colourspace.

References
----------
:cite:`Canon2014a`

CINEMA_GAMUT_COLOURSPACE : RGB_Colourspace
"""
