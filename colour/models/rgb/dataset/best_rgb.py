# -*- coding: utf-8 -*-
"""
Best RGB Colourspace
====================

Defines the *Best RGB* colourspace:

-   :attr:`colour.models.BEST_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`HutchColord` : HutchColor. (n.d.). BestRGB (4 K). Retrieved from
    http://www.hutchcolor.com/profiles/BestRGB.zip
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
    'BEST_RGB_PRIMARIES', 'BEST_RGB_WHITEPOINT_NAME', 'BEST_RGB_WHITEPOINT',
    'BEST_RGB_TO_XYZ_MATRIX', 'XYZ_TO_BEST_RGB_MATRIX', 'BEST_RGB_COLOURSPACE'
]

BEST_RGB_PRIMARIES = np.array([
    [0.735191637630662, 0.264808362369338],
    [0.215336134453781, 0.774159663865546],
    [0.130122950819672, 0.034836065573770],
])
"""
*Best RGB* colourspace primaries.

BEST_RGB_PRIMARIES : ndarray, (3, 2)
"""

BEST_RGB_WHITEPOINT_NAME = 'D50'
"""
*Best RGB* colourspace whitepoint name.

BEST_RGB_WHITEPOINT_NAME : unicode
"""

BEST_RGB_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    BEST_RGB_WHITEPOINT_NAME])
"""
*Best RGB* colourspace whitepoint.

BEST_RGB_WHITEPOINT : ndarray
"""

BEST_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(BEST_RGB_PRIMARIES,
                                                   BEST_RGB_WHITEPOINT)
"""
*Best RGB* colourspace to *CIE XYZ* tristimulus values matrix.

BEST_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_BEST_RGB_MATRIX = np.linalg.inv(BEST_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Best RGB* colourspace matrix.

XYZ_TO_BEST_RGB_MATRIX : array_like, (3, 3)
"""

BEST_RGB_COLOURSPACE = RGB_Colourspace(
    'Best RGB',
    BEST_RGB_PRIMARIES,
    BEST_RGB_WHITEPOINT,
    BEST_RGB_WHITEPOINT_NAME,
    BEST_RGB_TO_XYZ_MATRIX,
    XYZ_TO_BEST_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
BEST_RGB_COLOURSPACE.__doc__ = """
*Best RGB* colourspace.

References
----------
:cite:`HutchColord`

BEST_RGB_COLOURSPACE : RGB_Colourspace
"""
