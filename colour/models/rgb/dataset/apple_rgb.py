# -*- coding: utf-8 -*-
"""
Apple RGB Colourspace
=====================

Defines the *Apple RGB* colourspace:

-   :attr:`colour.models.APPLE_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Susstrunk1999a` : Susstrunk, S., Buckley, R., & Swen, S. (1999).
    Standard RGB Color Spaces. ISBN:2166-9635
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
    'APPLE_RGB_PRIMARIES', 'APPLE_RGB_WHITEPOINT_NAME', 'APPLE_RGB_WHITEPOINT',
    'APPLE_RGB_TO_XYZ_MATRIX', 'XYZ_TO_APPLE_RGB_MATRIX',
    'APPLE_RGB_COLOURSPACE'
]

APPLE_RGB_PRIMARIES = np.array([
    [0.6250, 0.3400],
    [0.2800, 0.5950],
    [0.1550, 0.0700],
])
"""
*Apple RGB* colourspace primaries.

APPLE_RGB_PRIMARIES : ndarray, (3, 2)
"""

APPLE_RGB_WHITEPOINT_NAME = 'D65'
"""
*Apple RGB* colourspace whitepoint name.

APPLE_RGB_WHITEPOINT_NAME : unicode
"""

APPLE_RGB_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    APPLE_RGB_WHITEPOINT_NAME])
"""
*Apple RGB* colourspace whitepoint.

APPLE_RGB_WHITEPOINT : ndarray
"""

APPLE_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(APPLE_RGB_PRIMARIES,
                                                    APPLE_RGB_WHITEPOINT)
"""
*Apple RGB* colourspace to *CIE XYZ* tristimulus values matrix.

APPLE_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_APPLE_RGB_MATRIX = np.linalg.inv(APPLE_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Apple RGB* colourspace matrix.

XYZ_TO_APPLE_RGB_MATRIX : array_like, (3, 3)
"""

APPLE_RGB_COLOURSPACE = RGB_Colourspace(
    'Apple RGB',
    APPLE_RGB_PRIMARIES,
    APPLE_RGB_WHITEPOINT,
    APPLE_RGB_WHITEPOINT_NAME,
    APPLE_RGB_TO_XYZ_MATRIX,
    XYZ_TO_APPLE_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 1.8),
    partial(gamma_function, exponent=1.8),
)
APPLE_RGB_COLOURSPACE.__doc__ = """
*Apple RGB* colourspace.

References
----------
:cite:`Susstrunk1999a`

APPLE_RGB_COLOURSPACE : RGB_Colourspace
"""
