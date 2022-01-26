# -*- coding: utf-8 -*-
"""
P3-D65 Colourspace
==================

Defines the *P3-D65* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_P3_D65`.
"""

from __future__ import annotations

import numpy as np
from functools import partial

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    gamma_function,
    normalised_primary_matrix,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_P3_D65',
    'WHITEPOINT_NAME_P3_D65',
    'CCS_WHITEPOINT_P3_D65',
    'MATRIX_P3_D65_TO_XYZ',
    'MATRIX_XYZ_TO_P3_D65',
    'RGB_COLOURSPACE_P3_D65',
]

PRIMARIES_P3_D65: NDArray = np.array([
    [0.6800, 0.3200],
    [0.2650, 0.6900],
    [0.1500, 0.0600],
])
"""
*P3-D65* colourspace primaries.
"""

WHITEPOINT_NAME_P3_D65: str = 'D65'
"""
*P3-D65* colourspace whitepoint name.
"""

CCS_WHITEPOINT_P3_D65: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_P3_D65])
"""
*P3-D65* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_P3_D65_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_P3_D65, CCS_WHITEPOINT_P3_D65)
"""
*P3-D65* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_P3_D65: NDArray = np.linalg.inv(MATRIX_P3_D65_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *P3-D65* colourspace matrix.
"""

RGB_COLOURSPACE_P3_D65: RGB_Colourspace = RGB_Colourspace(
    'P3-D65',
    PRIMARIES_P3_D65,
    CCS_WHITEPOINT_P3_D65,
    WHITEPOINT_NAME_P3_D65,
    MATRIX_P3_D65_TO_XYZ,
    MATRIX_XYZ_TO_P3_D65,
    partial(gamma_function, exponent=1 / 2.6),
    partial(gamma_function, exponent=2.6),
)
RGB_COLOURSPACE_P3_D65.__doc__ = """
*P3-D65* colourspace.
"""
