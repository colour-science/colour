# -*- coding: utf-8 -*-
"""
DaVinci Wide Gamut Colourspace
==============================

Defines the *DaVinci Wide Gamut* *RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT`.

References
----------
-   :cite:`BlackmagicDesign2020` : Blackmagic Design. (2020).
    DaVinci Wide Gamut - DaVinci Resolve Studio 17 Public Beta 1.
-   :cite:`BlackmagicDesign2020a` : Blackmagic Design. (2020). Wide Gamut
    Intermediate DaVinci Resolve. Retrieved December 12, 2020, from
    https://documents.blackmagicdesign.com/InformationNotes/\
DaVinci_Resolve_17_Wide_Gamut_Intermediate.pdf?_v=1607414410000
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    oetf_DaVinciIntermediate,
    oetf_inverse_DaVinciIntermediate,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_DAVINCI_WIDE_GAMUT',
    'WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT',
    'CCS_WHITEPOINT_DAVINCI_WIDE_GAMUT',
    'MATRIX_DAVINCI_WIDE_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_DAVINCI_WIDE_GAMUT',
    'RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT',
]

PRIMARIES_DAVINCI_WIDE_GAMUT: NDArray = np.array([
    [0.8000, 0.3130],
    [0.1682, 0.9877],
    [0.0790, -0.1155],
])
"""
*DaVinci Wide Gamut* colourspace primaries.
"""

WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT: str = 'D65'
"""
*DaVinci Wide Gamut* colourspace whitepoint name.
"""

CCS_WHITEPOINT_DAVINCI_WIDE_GAMUT: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT])
"""
*DaVinci Wide Gamut* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_DAVINCI_WIDE_GAMUT_TO_XYZ: NDArray = np.array([
    [0.70062239, 0.14877482, 0.10105872],
    [0.27411851, 0.87363190, -0.14775041],
    [-0.09896291, -0.13789533, 1.32591599],
])
"""
*DaVinci Wide Gamut* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_DAVINCI_WIDE_GAMUT: NDArray = np.array([
    [1.51667204, -0.28147805, -0.14696363],
    [-0.46491710, 1.25142378, 0.17488461],
    [0.06484905, 0.10913934, 0.76141462],
])
"""
*CIE XYZ* tristimulus values to *DaVinci Wide Gamut* colourspace matrix.
"""

RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT: RGB_Colourspace = RGB_Colourspace(
    'DaVinci Wide Gamut',
    PRIMARIES_DAVINCI_WIDE_GAMUT,
    CCS_WHITEPOINT_DAVINCI_WIDE_GAMUT,
    WHITEPOINT_NAME_DAVINCI_WIDE_GAMUT,
    MATRIX_DAVINCI_WIDE_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_DAVINCI_WIDE_GAMUT,
    oetf_DaVinciIntermediate,
    oetf_inverse_DaVinciIntermediate,
    use_derived_matrix_RGB_to_XYZ=True,
    use_derived_matrix_XYZ_to_RGB=True,
)
RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT.__doc__ = """
*DaVinci Wide Gamut* colourspace.

References
----------
:cite:`BlackmagicDesign2020`, :cite:`BlackmagicDesign2020a`
"""
