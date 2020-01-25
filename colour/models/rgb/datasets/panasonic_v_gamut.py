# -*- coding: utf-8 -*-
"""
Panasonic V-Gamut Colourspace
=============================

Defines the *Panasonic V-Gamut* colourspace:

-   :attr:`colour.models.V_GAMUT_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Panasonic2014a` : Panasonic. (2014). VARICAM V-Log/V-Gamut.
    Retrieved from http://pro-av.panasonic.net/en/varicam/common/pdf/\
VARICAM_V-Log_V-Gamut.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_encoding_VLog,
                               log_decoding_VLog)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'V_GAMUT_PRIMARIES', 'V_GAMUT_WHITEPOINT_NAME', 'V_GAMUT_WHITEPOINT',
    'V_GAMUT_TO_XYZ_MATRIX', 'XYZ_TO_V_GAMUT_MATRIX', 'V_GAMUT_COLOURSPACE'
]

V_GAMUT_PRIMARIES = np.array([
    [0.7300, 0.2800],
    [0.1650, 0.8400],
    [0.1000, -0.0300],
])
"""
*Panasonic V-Gamut* colourspace primaries.

V_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

V_GAMUT_WHITEPOINT_NAME = 'D65'
"""
*Panasonic V-Gamut* colourspace whitepoint name.

V_GAMUT_WHITEPOINT : unicode
"""

V_GAMUT_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    V_GAMUT_WHITEPOINT_NAME])
"""
*Panasonic V-Gamut* colourspace whitepoint.

V_GAMUT_WHITEPOINT : ndarray
"""

V_GAMUT_TO_XYZ_MATRIX = np.array([
    [0.679644, 0.152211, 0.118600],
    [0.260686, 0.774894, -0.035580],
    [-0.009310, -0.004612, 1.102980],
])
"""
*Panasonic V-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

V_GAMUT_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_V_GAMUT_MATRIX = np.array([
    [1.589012, -0.313204, -0.180965],
    [-0.534053, 1.396011, 0.102458],
    [0.011179, 0.003194, 0.905535],
])
"""
*CIE XYZ* tristimulus values to *Panasonic V-Gamut* colourspace matrix.

XYZ_TO_V_GAMUT_MATRIX : array_like, (3, 3)
"""

V_GAMUT_COLOURSPACE = RGB_Colourspace(
    'V-Gamut',
    V_GAMUT_PRIMARIES,
    V_GAMUT_WHITEPOINT,
    V_GAMUT_WHITEPOINT_NAME,
    V_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_V_GAMUT_MATRIX,
    log_encoding_VLog,
    log_decoding_VLog,
)
V_GAMUT_COLOURSPACE.__doc__ = """
*Panasonic V-Gamut* colourspace.

References
----------
:cite:`Panasonic2014a`

V_GAMUT_COLOURSPACE : RGB_Colourspace
"""
