# -*- coding: utf-8 -*-
"""
Display P3 Colourspace
============================

Defines the *Display P3* colourspace:

-   :attr:`colour.models.DISPLAY_P3_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Apple2019` : Apple. (2019). https://developer.apple.com/\
    documentation/coregraphics/cgcolorspace/1408916-displayp3#discussion
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, eotf_inverse_sRGB, eotf_sRGB,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'DISPLAY_P3_PRIMARIES', 'DISPLAY_P3_WHITEPOINT_NAME',
    'DISPLAY_P3_WHITEPOINT', 'DISPLAY_P3_TO_XYZ_MATRIX',
    'XYZ_TO_DISPLAY_P3_MATRIX', 'DISPLAY_P3_COLOURSPACE'
]

DISPLAY_P3_PRIMARIES = np.array([
    [0.6800, 0.3200],
    [0.2650, 0.6900],
    [0.1500, 0.0600],
])
"""
*Display P3* colourspace primaries.

DISPLAY_P3_PRIMARIES : ndarray, (3, 2)
"""

DISPLAY_P3_WHITEPOINT_NAME = 'D65'
"""
*Display P3* colourspace whitepoint name.

DISPLAY_P3_WHITEPOINT : unicode
"""

DISPLAY_P3_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
        DISPLAY_P3_WHITEPOINT_NAME])
"""
*Display P3* colourspace whitepoint.

DISPLAY_P3_WHITEPOINT : ndarray
"""

DISPLAY_P3_TO_XYZ_MATRIX = \
    normalised_primary_matrix(DISPLAY_P3_PRIMARIES, DISPLAY_P3_WHITEPOINT)
"""
*Display P3* colourspace to *CIE XYZ* tristimulus values matrix.

DISPLAY_P3_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DISPLAY_P3_MATRIX = np.linalg.inv(DISPLAY_P3_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Display P3* colourspace matrix.

XYZ_TO_DISPLAY_P3_MATRIX : array_like, (3, 3)
"""


DISPLAY_P3_COLOURSPACE = RGB_Colourspace(
    'Display P3',
    DISPLAY_P3_PRIMARIES,
    DISPLAY_P3_WHITEPOINT,
    DISPLAY_P3_WHITEPOINT_NAME,
    DISPLAY_P3_TO_XYZ_MATRIX,
    XYZ_TO_DISPLAY_P3_MATRIX,
    eotf_inverse_sRGB,
    eotf_sRGB,
)
DISPLAY_P3_COLOURSPACE.__doc__ = """
*Display P3* colourspace.

References
----------
:cite:`Apple2019`

DISPLAY_P3_COLOURSPACE : RGB_Colourspace
"""
