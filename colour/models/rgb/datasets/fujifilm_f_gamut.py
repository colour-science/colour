# -*- coding: utf-8 -*-
"""
Fujifilm F-Gamut Colourspace
============================

Defines the *Fujifilm F-Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_F_GAMUT`.

References
----------
-   :cite:`Fujifilm2016` : Fujifilm. (2016). F-Log Data Sheet Ver.1.0 (pp.
    1-4). https://www.fujifilm.com/support/digital_cameras/software/lut/pdf/\
F-Log_DataSheet_E_Ver.1.0.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    log_encoding_FLog,
    normalised_primary_matrix,
    log_decoding_FLog,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_F_GAMUT',
    'WHITEPOINT_NAME_F_GAMUT',
    'CCS_WHITEPOINT_F_GAMUT',
    'MATRIX_F_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_F_GAMUT',
    'RGB_COLOURSPACE_F_GAMUT',
]

PRIMARIES_F_GAMUT: NDArray = np.array([
    [0.70800, 0.29200],
    [0.17000, 0.79700],
    [0.13100, 0.04600],
])
"""
*Fujifilm F-Gamut* colourspace primaries.
"""

WHITEPOINT_NAME_F_GAMUT: str = 'D65'
"""
*Fujifilm F-Gamut* colourspace whitepoint name.
"""

CCS_WHITEPOINT_F_GAMUT: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_F_GAMUT])
"""
*Fujifilm F-Gamut* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_F_GAMUT_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_F_GAMUT, CCS_WHITEPOINT_F_GAMUT)
"""
*Fujifilm F-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_F_GAMUT: NDArray = np.linalg.inv(MATRIX_F_GAMUT_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Fujifilm F-Gamut* colourspace matrix.
"""

RGB_COLOURSPACE_F_GAMUT: RGB_Colourspace = RGB_Colourspace(
    'F-Gamut',
    PRIMARIES_F_GAMUT,
    CCS_WHITEPOINT_F_GAMUT,
    WHITEPOINT_NAME_F_GAMUT,
    MATRIX_F_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_F_GAMUT,
    log_encoding_FLog,
    log_decoding_FLog,
)
RGB_COLOURSPACE_F_GAMUT.__doc__ = """
*Fujifilm F-Gamut* colourspace.

References
----------
:cite:`Fujifilm2016`
"""
