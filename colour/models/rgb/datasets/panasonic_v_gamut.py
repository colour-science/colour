# -*- coding: utf-8 -*-
"""
Panasonic V-Gamut Colourspace
=============================

Defines the *Panasonic V-Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_V_GAMUT`.

References
----------
-   :cite:`Panasonic2014a` : Panasonic. (2014). VARICAM V-Log/V-Gamut (pp.
    1-7).
    http://pro-av.panasonic.net/en/varicam/common/pdf/VARICAM_V-Log_V-Gamut.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    log_encoding_VLog,
    log_decoding_VLog,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_V_GAMUT',
    'WHITEPOINT_NAME_V_GAMUT',
    'CCS_WHITEPOINT_V_GAMUT',
    'MATRIX_V_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_V_GAMUT',
    'RGB_COLOURSPACE_V_GAMUT',
]

PRIMARIES_V_GAMUT: NDArray = np.array([
    [0.7300, 0.2800],
    [0.1650, 0.8400],
    [0.1000, -0.0300],
])
"""
*Panasonic V-Gamut* colourspace primaries.
"""

WHITEPOINT_NAME_V_GAMUT: str = 'D65'
"""
*Panasonic V-Gamut* colourspace whitepoint name.
"""

CCS_WHITEPOINT_V_GAMUT: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_V_GAMUT])
"""
*Panasonic V-Gamut* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_V_GAMUT_TO_XYZ: NDArray = np.array([
    [0.679644, 0.152211, 0.118600],
    [0.260686, 0.774894, -0.035580],
    [-0.009310, -0.004612, 1.102980],
])
"""
*Panasonic V-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_V_GAMUT: NDArray = np.array([
    [1.589012, -0.313204, -0.180965],
    [-0.534053, 1.396011, 0.102458],
    [0.011179, 0.003194, 0.905535],
])
"""
*CIE XYZ* tristimulus values to *Panasonic V-Gamut* colourspace matrix.
"""

RGB_COLOURSPACE_V_GAMUT: RGB_Colourspace = RGB_Colourspace(
    'V-Gamut',
    PRIMARIES_V_GAMUT,
    CCS_WHITEPOINT_V_GAMUT,
    WHITEPOINT_NAME_V_GAMUT,
    MATRIX_V_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_V_GAMUT,
    log_encoding_VLog,
    log_decoding_VLog,
)
RGB_COLOURSPACE_V_GAMUT.__doc__ = """
*Panasonic V-Gamut* colourspace.

References
----------
:cite:`Panasonic2014a`
"""
