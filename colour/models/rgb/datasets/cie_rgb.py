# -*- coding: utf-8 -*-
"""
CIE RGB Colourspace
===================

Defines the *CIE RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_CIE_RGB`.

References
----------
-   :cite:`Fairman1997` : Fairman, H. S., Brill, M. H., & Hemmendinger, H.
    (1997). How the CIE 1931 color-matching functions were derived from
    Wright-Guild data. Color Research & Application, 22(1), 11-23.
    doi:10.1002/(SICI)1520-6378(199702)22:1<11::AID-COL4>3.0.CO;2-7
"""

from __future__ import annotations

import numpy as np
from functools import partial

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import RGB_Colourspace, gamma_function

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_CIE_RGB',
    'WHITEPOINT_NAME_CIE_RGB',
    'CCS_WHITEPOINT_CIE_RGB',
    'MATRIX_CIE_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_CIE_RGB',
    'RGB_COLOURSPACE_CIE_RGB',
]

PRIMARIES_CIE_RGB: NDArray = np.array([
    [0.734742840005998, 0.265257159994002],
    [0.273779033824958, 0.717477700256116],
    [0.166555629580280, 0.008910726182545],
])
"""
*CIE RGB* colourspace primaries.

Notes
-----
-   *CIE RGB* colourspace primaries were computed using
    :attr:`colour.models.rgb.datasets.cie_rgb.MATRIX_CIE_RGB_TO_XYZ` attribute
    and :func:`colour.primaries_whitepoint` definition.
"""

WHITEPOINT_NAME_CIE_RGB: str = 'E'
"""
*CIE RGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_CIE_RGB: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_CIE_RGB])
"""
*CIE RGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_CIE_RGB_TO_XYZ: NDArray = np.array([
    [0.4900, 0.3100, 0.2000],
    [0.1769, 0.8124, 0.0107],
    [0.0000, 0.0099, 0.9901],
])
"""
*CIE RGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_CIE_RGB: NDArray = np.linalg.inv(MATRIX_CIE_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *CIE RGB* colourspace matrix.
"""

RGB_COLOURSPACE_CIE_RGB: RGB_Colourspace = RGB_Colourspace(
    'CIE RGB',
    PRIMARIES_CIE_RGB,
    CCS_WHITEPOINT_CIE_RGB,
    WHITEPOINT_NAME_CIE_RGB,
    MATRIX_CIE_RGB_TO_XYZ,
    MATRIX_XYZ_TO_CIE_RGB,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_CIE_RGB.__doc__ = """
*CIE RGB* colourspace.

References
----------
:cite:`Fairman1997`
"""
