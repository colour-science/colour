# -*- coding: utf-8 -*-
"""
Beta RGB Colourspace
====================

Defines the *Beta RGB* colourspace:

-   :attr:`colour.models.BETA_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Lindbloom2014a` : Lindbloom, B. (2014). RGB Working Space
    Information. Retrieved April 11, 2014, from
    http://www.brucelindbloom.com/WorkingSpaceInfo.html
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
    'BETA_RGB_PRIMARIES', 'BETA_RGB_WHITEPOINT_NAME', 'BETA_RGB_WHITEPOINT',
    'BETA_RGB_TO_XYZ_MATRIX', 'XYZ_TO_BETA_RGB_MATRIX', 'BETA_RGB_COLOURSPACE'
]

BETA_RGB_PRIMARIES = np.array([
    [0.6888, 0.3112],
    [0.1986, 0.7551],
    [0.1265, 0.0352],
])
"""
*Beta RGB* colourspace primaries.

BETA_RGB_PRIMARIES : ndarray, (3, 2)
"""

BETA_RGB_WHITEPOINT_NAME = 'D50'
"""
*Beta RGB* colourspace whitepoint name.

BETA_RGB_WHITEPOINT_NAME : unicode
"""

BETA_RGB_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    BETA_RGB_WHITEPOINT_NAME])
"""
*Beta RGB* colourspace whitepoint.

BETA_RGB_WHITEPOINT : ndarray
"""

BETA_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(BETA_RGB_PRIMARIES,
                                                   BETA_RGB_WHITEPOINT)
"""
*Beta RGB* colourspace to *CIE XYZ* tristimulus values matrix.

BETA_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_BETA_RGB_MATRIX = np.linalg.inv(BETA_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Beta RGB* colourspace matrix.

XYZ_TO_BETA_RGB_MATRIX : array_like, (3, 3)
"""

BETA_RGB_COLOURSPACE = RGB_Colourspace(
    'Beta RGB',
    BETA_RGB_PRIMARIES,
    BETA_RGB_WHITEPOINT,
    BETA_RGB_WHITEPOINT_NAME,
    BETA_RGB_TO_XYZ_MATRIX,
    XYZ_TO_BETA_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
BETA_RGB_COLOURSPACE.__doc__ = """
*Beta RGB* colourspace.

References
----------
:cite:`Lindbloom2014a`

BETA_RGB_COLOURSPACE : RGB_Colourspace
"""
