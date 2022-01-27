# -*- coding: utf-8 -*-
"""
Xtreme RGB Colourspace
======================

Defines the *Xtreme RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_XTREME_RGB`.

References
----------
-   :cite:`HutchColore` : HutchColor. (n.d.). XtremeRGB (4 K).
    http://www.hutchcolor.com/profiles/XtremeRGB.zip
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
    'PRIMARIES_XTREME_RGB',
    'WHITEPOINT_NAME_XTREME_RGB',
    'CCS_WHITEPOINT_XTREME_RGB',
    'MATRIX_XTREME_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_XTREME_RGB',
    'RGB_COLOURSPACE_XTREME_RGB',
]

PRIMARIES_XTREME_RGB: NDArray = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
])
"""
*Xtreme RGB* colourspace primaries.
"""

WHITEPOINT_NAME_XTREME_RGB: str = 'D50'
"""
*Xtreme RGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_XTREME_RGB: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_XTREME_RGB])
"""
*Xtreme RGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_XTREME_RGB_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_XTREME_RGB, CCS_WHITEPOINT_XTREME_RGB)
"""
*Xtreme RGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_XTREME_RGB: NDArray = np.linalg.inv(MATRIX_XTREME_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Xtreme RGB* colourspace matrix.
"""

RGB_COLOURSPACE_XTREME_RGB: RGB_Colourspace = RGB_Colourspace(
    'Xtreme RGB',
    PRIMARIES_XTREME_RGB,
    CCS_WHITEPOINT_XTREME_RGB,
    WHITEPOINT_NAME_XTREME_RGB,
    MATRIX_XTREME_RGB_TO_XYZ,
    MATRIX_XYZ_TO_XTREME_RGB,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_XTREME_RGB.__doc__ = """
*Xtreme RGB* colourspace.

References
----------
:cite:`HutchColore`
"""
