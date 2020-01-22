# -*- coding: utf-8 -*-
"""
P3-D65 Colourspace
==================

Defines the *P3-D65* colourspace:

-   :attr:`colour.models.P3_D65_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, gamma_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'P3_D65_PRIMARIES', 'P3_D65_WHITEPOINT_NAME', 'P3_D65_WHITEPOINT',
    'P3_D65_TO_XYZ_MATRIX', 'XYZ_TO_P3_D65_MATRIX', 'P3_D65_COLOURSPACE'
]

P3_D65_PRIMARIES = np.array([
    [0.6800, 0.3200],
    [0.2650, 0.6900],
    [0.1500, 0.0600],
])
"""
*P3-D65* colourspace primaries.

P3_D65_PRIMARIES : ndarray, (3, 2)
"""

P3_D65_WHITEPOINT_NAME = 'D65'
"""
*P3-D65* colourspace whitepoint name.

P3_D65_WHITEPOINT_NAME : unicode
"""

P3_D65_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][P3_D65_WHITEPOINT_NAME])
"""
*P3-D65* colourspace whitepoint.

P3_D65_WHITEPOINT : ndarray
"""

P3_D65_TO_XYZ_MATRIX = normalised_primary_matrix(P3_D65_PRIMARIES,
                                                 P3_D65_WHITEPOINT)
"""
*P3-D65* colourspace to *CIE XYZ* tristimulus values matrix.

P3_D65_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_P3_D65_MATRIX = np.linalg.inv(P3_D65_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *P3-D65* colourspace matrix.

XYZ_TO_P3_D65_MATRIX : array_like, (3, 3)
"""

P3_D65_COLOURSPACE = RGB_Colourspace(
    'P3-D65',
    P3_D65_PRIMARIES,
    P3_D65_WHITEPOINT,
    P3_D65_WHITEPOINT_NAME,
    P3_D65_TO_XYZ_MATRIX,
    XYZ_TO_P3_D65_MATRIX,
    partial(gamma_function, exponent=1 / 2.6),
    partial(gamma_function, exponent=2.6),
)
P3_D65_COLOURSPACE.__doc__ = """
*P3-D65* colourspace.

P3_D65_COLOURSPACE : RGB_Colourspace
"""
