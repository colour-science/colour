# -*- coding: utf-8 -*-
"""
DJI D-Gamut Colourspace
=======================

Defines the *DJI D-Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_DJI_D_GAMUT`.

References
----------
-   :cite:`DJI2017` : Dji. (2017). White Paper on D-Log and D-Gamut of DJI
    Cinema Color System (pp. 1-5).
    https://dl.djicdn.com/downloads/zenmuse+x7/20171010/\
D-Log_D-Gamut_Whitepaper.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    log_encoding_DJIDLog,
    log_decoding_DJIDLog,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_DJI_D_GAMUT',
    'WHITEPOINT_NAME_DJI_D_GAMUT',
    'CCS_WHITEPOINT_DJI_D_GAMUT',
    'MATRIX_DJI_D_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_DJI_D_GAMUT',
    'RGB_COLOURSPACE_DJI_D_GAMUT',
]

PRIMARIES_DJI_D_GAMUT: NDArray = np.array([
    [0.71, 0.31],
    [0.21, 0.88],
    [0.09, -0.08],
])
"""
*DJI D-Gamut* colourspace primaries.
"""

WHITEPOINT_NAME_DJI_D_GAMUT: str = 'D65'
"""
*DJI D-Gamut* colourspace whitepoint name.
"""

CCS_WHITEPOINT_DJI_D_GAMUT: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_DJI_D_GAMUT])
"""
*DJI D-Gamut* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_DJI_D_GAMUT_TO_XYZ: NDArray = np.array([
    [0.6482, 0.1940, 0.1082],
    [0.2830, 0.8132, -0.0962],
    [-0.0183, -0.0832, 1.1903],
])
"""
*DJI D-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_DJI_D_GAMUT: NDArray = np.array([
    [1.7257, -0.4314, -0.1917],
    [-0.6025, 1.3906, 0.1671],
    [-0.0156, 0.0905, 0.8489],
])
"""
*CIE XYZ* tristimulus values to *DJI D-Gamut* colourspace matrix.
"""

RGB_COLOURSPACE_DJI_D_GAMUT: RGB_Colourspace = RGB_Colourspace(
    'DJI D-Gamut',
    PRIMARIES_DJI_D_GAMUT,
    CCS_WHITEPOINT_DJI_D_GAMUT,
    WHITEPOINT_NAME_DJI_D_GAMUT,
    MATRIX_DJI_D_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_DJI_D_GAMUT,
    log_encoding_DJIDLog,
    log_decoding_DJIDLog,
)
RGB_COLOURSPACE_DJI_D_GAMUT.__doc__ = """
*DJI_D-Gamut* colourspace.

References
----------
:cite:`DJI2017`
"""
