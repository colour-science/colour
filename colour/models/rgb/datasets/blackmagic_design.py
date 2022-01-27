# -*- coding: utf-8 -*-
"""
Blackmagic Design Colourspaces
==============================

Defines the *Blackmagic Design* *RGB* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT`.

References
----------
-   :cite:`BlackmagicDesign2021` : Blackmagic Design. (2021). Blackmagic
    Generation 5 Color Science. https://drive.google.com/file/d/\
1FF5WO2nvI9GEWb4_EntrBoV9ZIuFToZd/view
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    oetf_BlackmagicFilmGeneration5,
    oetf_inverse_BlackmagicFilmGeneration5,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_BLACKMAGIC_WIDE_GAMUT',
    'WHITEPOINT_NAME_BLACKMAGIC_WIDE_GAMUT',
    'CCS_WHITEPOINT_BLACKMAGIC_WIDE_GAMUT',
    'MATRIX_BLACKMAGIC_WIDE_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_BLACKMAGIC_WIDE_GAMUT',
    'RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT',
]

PRIMARIES_BLACKMAGIC_WIDE_GAMUT: NDArray = np.array([
    [0.7177215, 0.3171181],
    [0.2280410, 0.8615690],
    [0.1005841, -0.0820452],
])
"""
*Blackmagic Wide Gamut* colourspace primaries.
"""

WHITEPOINT_NAME_BLACKMAGIC_WIDE_GAMUT: str = 'Blackmagic Wide Gamut'
"""
*Blackmagic Wide Gamut* colourspace whitepoint name.

Notes
-----
-   *Blackmagic Wide Gamut* colourspace whitepoint is an uncommonly rounded
    *D65* variant at 7 decimals: [0.3127170, 0.3290312]
"""

CCS_WHITEPOINT_BLACKMAGIC_WIDE_GAMUT: NDArray = (
    CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
        WHITEPOINT_NAME_BLACKMAGIC_WIDE_GAMUT])
"""
*Blackmagic Wide Gamut* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_BLACKMAGIC_WIDE_GAMUT_TO_XYZ: NDArray = np.array([
    [0.606530, 0.220408, 0.123479],
    [0.267989, 0.832731, -0.100720],
    [-0.029442, -0.086611, 1.204861],
])
"""
*Blackmagic Wide Gamut* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_BLACKMAGIC_WIDE_GAMUT: NDArray = np.array([
    [1.866382, -0.518397, -0.234610],
    [-0.600342, 1.378149, 0.176732],
    [0.002452, 0.086400, 0.836943],
])
"""
*CIE XYZ* tristimulus values to *Blackmagic Wide Gamut* colourspace matrix.
"""

RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT: RGB_Colourspace = RGB_Colourspace(
    'Blackmagic Wide Gamut',
    PRIMARIES_BLACKMAGIC_WIDE_GAMUT,
    CCS_WHITEPOINT_BLACKMAGIC_WIDE_GAMUT,
    WHITEPOINT_NAME_BLACKMAGIC_WIDE_GAMUT,
    MATRIX_BLACKMAGIC_WIDE_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_BLACKMAGIC_WIDE_GAMUT,
    oetf_BlackmagicFilmGeneration5,
    oetf_inverse_BlackmagicFilmGeneration5,
    use_derived_matrix_RGB_to_XYZ=True,
    use_derived_matrix_XYZ_to_RGB=True,
)
RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT.__doc__ = """
*Blackmagic Wide Gamut* colourspace.

References
----------
:cite:`BlackmagicDesign2021`
"""
