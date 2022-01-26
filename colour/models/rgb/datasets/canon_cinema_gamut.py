# -*- coding: utf-8 -*-
"""
Canon Cinema Gamut Colourspace
==============================

Defines the *Canon Cinema Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_CINEMA_GAMUT`.

References
----------
-   :cite:`Canon2014a` : Canon. (2014). EOS C500 Firmware Update. Retrieved
    August 27, 2016, from
    https://www.usa.canon.com/internet/portal/us/home/explore/\
product-showcases/cameras-and-lenses/cinema-eos-firmware/c500
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    linear_function,
    normalised_primary_matrix,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_CINEMA_GAMUT',
    'WHITEPOINT_NAME_CINEMA_GAMUT',
    'CCS_WHITEPOINT_CINEMA_GAMUT',
    'MATRIX_CINEMA_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_CINEMA_GAMUT',
    'RGB_COLOURSPACE_CINEMA_GAMUT',
]

PRIMARIES_CINEMA_GAMUT: NDArray = np.array([
    [0.7400, 0.2700],
    [0.1700, 1.1400],
    [0.0800, -0.1000],
])
"""
*Canon Cinema Gamut* colourspace primaries.
"""

WHITEPOINT_NAME_CINEMA_GAMUT: str = 'D65'
"""
*Canon Cinema Gamut* colourspace whitepoint name.
"""

CCS_WHITEPOINT_CINEMA_GAMUT: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_CINEMA_GAMUT])
"""
*Canon Cinema Gamut* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_CINEMA_GAMUT_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_CINEMA_GAMUT, CCS_WHITEPOINT_CINEMA_GAMUT)
"""
*Canon Cinema Gamut* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_CINEMA_GAMUT: NDArray = np.linalg.inv(MATRIX_CINEMA_GAMUT_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Canon Cinema Gamut* colourspace matrix.
"""

RGB_COLOURSPACE_CINEMA_GAMUT: RGB_Colourspace = RGB_Colourspace(
    'Cinema Gamut',
    PRIMARIES_CINEMA_GAMUT,
    CCS_WHITEPOINT_CINEMA_GAMUT,
    WHITEPOINT_NAME_CINEMA_GAMUT,
    MATRIX_CINEMA_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_CINEMA_GAMUT,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_CINEMA_GAMUT.__doc__ = """
*Canon Cinema Gamut* colourspace.

References
----------
:cite:`Canon2014a`
"""
