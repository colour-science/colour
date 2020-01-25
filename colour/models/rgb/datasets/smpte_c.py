# -*- coding: utf-8 -*-
"""
SMPTE C Colourspace
===================

Defines the *SMPTE C* colourspace:

-   :attr:`SMPTE_C_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`SocietyofMotionPictureandTelevisionEngineers2004a` : Society of
Motion Picture and Television Engineers. (2004). RP 145:2004: SMPTE C Color
Monitor Colorimetry. RP 145:2004 (Vol. RP 145:200). The Society of Motion
Picture and Television Engineers. doi:10.5594/S9781614821649
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
    'SMPTE_C_PRIMARIES', 'SMPTE_C_WHITEPOINT_NAME', 'SMPTE_C_WHITEPOINT',
    'SMPTE_C_TO_XYZ_MATRIX', 'XYZ_TO_SMPTE_C_MATRIX', 'SMPTE_C_COLOURSPACE'
]

SMPTE_C_PRIMARIES = np.array([
    [0.630, 0.340],
    [0.310, 0.595],
    [0.155, 0.070],
])
"""
*SMPTE C* colourspace primaries.

SMPTE_C_PRIMARIES : ndarray, (3, 2)
"""

SMPTE_C_WHITEPOINT_NAME = 'D65'
"""
*SMPTE C* colourspace whitepoint name.

SMPTE_C_WHITEPOINT_NAME : unicode
"""

SMPTE_C_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    SMPTE_C_WHITEPOINT_NAME])
"""
*SMPTE C* colourspace whitepoint.

SMPTE_C_WHITEPOINT : ndarray
"""

SMPTE_C_TO_XYZ_MATRIX = normalised_primary_matrix(SMPTE_C_PRIMARIES,
                                                  SMPTE_C_WHITEPOINT)
"""
*SMPTE C* colourspace to *CIE XYZ* tristimulus values matrix.

SMPTE_C_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_SMPTE_C_MATRIX = np.linalg.inv(SMPTE_C_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *SMPTE C* colourspace matrix.

XYZ_TO_SMPTE_C_MATRIX : array_like, (3, 3)
"""

SMPTE_C_COLOURSPACE = RGB_Colourspace(
    'SMPTE C',
    SMPTE_C_PRIMARIES,
    SMPTE_C_WHITEPOINT,
    SMPTE_C_WHITEPOINT_NAME,
    SMPTE_C_TO_XYZ_MATRIX,
    XYZ_TO_SMPTE_C_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
"""
*SMPTE C* colourspace.

References
----------
:cite:`SocietyofMotionPictureandTelevisionEngineers2004a`

SMPTE_C_COLOURSPACE : RGB_Colourspace
"""
