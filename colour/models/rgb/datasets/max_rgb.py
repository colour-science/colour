# -*- coding: utf-8 -*-
"""
Max RGB Colourspace
===================

Defines the *Max RGB* colourspace:

-   :attr:`colour.models.MAX_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`HutchColorf` : HutchColor. (n.d.). MaxRGB (4 K). Retrieved from
    http://www.hutchcolor.com/profiles/MaxRGB.zip
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
    'MAX_RGB_PRIMARIES', 'MAX_RGB_WHITEPOINT_NAME', 'MAX_RGB_WHITEPOINT',
    'MAX_RGB_TO_XYZ_MATRIX', 'XYZ_TO_MAX_RGB_MATRIX', 'MAX_RGB_COLOURSPACE'
]

MAX_RGB_PRIMARIES = np.array([
    [0.73413379, 0.26586621],
    [0.10039113, 0.89960887],
    [0.03621495, 0.00000000],
])
"""
*Max RGB* colourspace primaries.

MAX_RGB_PRIMARIES : ndarray, (3, 2)
"""

MAX_RGB_WHITEPOINT_NAME = 'D50'
"""
*Max RGB* colourspace whitepoint name.

MAX_RGB_WHITEPOINT_NAME : unicode
"""

MAX_RGB_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    MAX_RGB_WHITEPOINT_NAME])
"""
*Max RGB* colourspace whitepoint.

MAX_RGB_WHITEPOINT : ndarray
"""

MAX_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(MAX_RGB_PRIMARIES,
                                                  MAX_RGB_WHITEPOINT)
"""
*Max RGB* colourspace to *CIE XYZ* tristimulus values matrix.

MAX_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_MAX_RGB_MATRIX = np.linalg.inv(MAX_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Max RGB* colourspace matrix.

XYZ_TO_MAX_RGB_MATRIX : array_like, (3, 3)
"""

MAX_RGB_COLOURSPACE = RGB_Colourspace(
    'Max RGB',
    MAX_RGB_PRIMARIES,
    MAX_RGB_WHITEPOINT,
    MAX_RGB_WHITEPOINT_NAME,
    MAX_RGB_TO_XYZ_MATRIX,
    XYZ_TO_MAX_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
MAX_RGB_COLOURSPACE.__doc__ = """
*Max RGB* colourspace.

References
----------
:cite:`HutchColorf`

MAX_RGB_COLOURSPACE : RGB_Colourspace
"""
