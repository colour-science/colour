# -*- coding: utf-8 -*-
"""
Adobe RGB (1998) Colourspace
============================

Defines the *Adobe RGB (1998)* *RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_ADOBE_RGB1998`.

References
----------
-   :cite:`AdobeSystems2005a` : Adobe Systems. (2005). Adobe RGB (1998) Color
    Image Encoding. http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
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
    'PRIMARIES_ADOBE_RGB1998',
    'WHITEPOINT_NAME_ADOBE_RGB1998',
    'CCS_WHITEPOINT_ADOBE_RGB1998',
    'MATRIX_ADOBE_RGB1998_TO_XYZ',
    'MATRIX_XYZ_TO_ADOBE_RGB1998',
    'RGB_COLOURSPACE_ADOBE_RGB1998',
]

PRIMARIES_ADOBE_RGB1998: NDArray = np.array([
    [0.6400, 0.3300],
    [0.2100, 0.7100],
    [0.1500, 0.0600],
])
"""
*Adobe RGB (1998)* colourspace primaries.
"""

WHITEPOINT_NAME_ADOBE_RGB1998: str = 'D65'
"""
*Adobe RGB (1998)* colourspace whitepoint name.
"""

CCS_WHITEPOINT_ADOBE_RGB1998: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_ADOBE_RGB1998])
"""
*Adobe RGB (1998)* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_ADOBE_RGB1998_TO_XYZ: NDArray = np.array([
    [0.57667, 0.18556, 0.18823],
    [0.29734, 0.62736, 0.07529],
    [0.02703, 0.07069, 0.99134],
])
"""
*Adobe RGB (1998)* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_ADOBE_RGB1998: NDArray = np.array([
    [2.04159, -0.56501, -0.34473],
    [-0.96924, 1.87597, 0.04156],
    [0.01344, -0.11836, 1.01517],
])
"""
*CIE XYZ* tristimulus values to *Adobe RGB (1998)* colourspace matrix.
"""

RGB_COLOURSPACE_ADOBE_RGB1998: RGB_Colourspace = RGB_Colourspace(
    'Adobe RGB (1998)',
    PRIMARIES_ADOBE_RGB1998,
    CCS_WHITEPOINT_ADOBE_RGB1998,
    WHITEPOINT_NAME_ADOBE_RGB1998,
    MATRIX_ADOBE_RGB1998_TO_XYZ,
    MATRIX_XYZ_TO_ADOBE_RGB1998,
    partial(gamma_function, exponent=1 / (563 / 256)),
    partial(gamma_function, exponent=563 / 256),
)
RGB_COLOURSPACE_ADOBE_RGB1998.__doc__ = """
*Adobe RGB (1998)* colourspace.

References
----------
:cite:`AdobeSystems2005a`
"""
