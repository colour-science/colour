# -*- coding: utf-8 -*-
"""
Max RGB Colourspace
===================

Defines the *Max RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_MAX_RGB`.

References
----------
-   :cite:`HutchColorf` : HutchColor. (n.d.). MaxRGB (4 K).
    http://www.hutchcolor.com/profiles/MaxRGB.zip
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
    'PRIMARIES_MAX_RGB',
    'WHITEPOINT_NAME_MAX_RGB',
    'CCS_WHITEPOINT_MAX_RGB',
    'MATRIX_MAX_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_MAX_RGB',
    'RGB_COLOURSPACE_MAX_RGB',
]

PRIMARIES_MAX_RGB: NDArray = np.array([
    [0.73413379, 0.26586621],
    [0.10039113, 0.89960887],
    [0.03621495, 0.00000000],
])
"""
*Max RGB* colourspace primaries.
"""

WHITEPOINT_NAME_MAX_RGB: str = 'D50'
"""
*Max RGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_MAX_RGB: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_MAX_RGB])
"""
*Max RGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_MAX_RGB_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_MAX_RGB, CCS_WHITEPOINT_MAX_RGB)
"""
*Max RGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_MAX_RGB: NDArray = np.linalg.inv(MATRIX_MAX_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Max RGB* colourspace matrix.
"""

RGB_COLOURSPACE_MAX_RGB: RGB_Colourspace = RGB_Colourspace(
    'Max RGB',
    PRIMARIES_MAX_RGB,
    CCS_WHITEPOINT_MAX_RGB,
    WHITEPOINT_NAME_MAX_RGB,
    MATRIX_MAX_RGB_TO_XYZ,
    MATRIX_XYZ_TO_MAX_RGB,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_MAX_RGB.__doc__ = """
*Max RGB* colourspace.

References
----------
:cite:`HutchColorf`
"""
