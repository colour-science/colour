# -*- coding: utf-8 -*-
"""
sRGB Colourspace
================

Defines the *sRGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_sRGB`.

References
----------
-   :cite:`InternationalElectrotechnicalCommission1999a` : International
    Electrotechnical Commission. (1999). IEC 61966-2-1:1999 - Multimedia
    systems and equipment - Colour measurement and management - Part 2-1:
    Colour management - Default RGB colour space - sRGB (p. 51).
    https://webstore.iec.ch/publication/6169
-   :cite:`InternationalTelecommunicationUnion2015i` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.709-6 - Parameter
    values for the HDTV standards for production and international programme
    exchange BT Series Broadcasting service (pp. 1-32).
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import RGB_Colourspace, eotf_inverse_sRGB, eotf_sRGB

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_sRGB',
    'WHITEPOINT_NAME_sRGB',
    'CCS_WHITEPOINT_sRGB',
    'MATRIX_sRGB_TO_XYZ',
    'MATRIX_XYZ_TO_sRGB',
    'RGB_COLOURSPACE_sRGB',
]

PRIMARIES_sRGB: NDArray = np.array([
    [0.6400, 0.3300],
    [0.3000, 0.6000],
    [0.1500, 0.0600],
])
"""
*sRGB* colourspace primaries.
"""

WHITEPOINT_NAME_sRGB: str = 'D65'
"""
*sRGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_sRGB: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_sRGB])
"""
*sRGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_sRGB_TO_XYZ: NDArray = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
])
"""
*sRGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_sRGB: NDArray = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])
"""
*CIE XYZ* tristimulus values to *sRGB* colourspace matrix.
"""

RGB_COLOURSPACE_sRGB: RGB_Colourspace = RGB_Colourspace(
    'sRGB',
    PRIMARIES_sRGB,
    CCS_WHITEPOINT_sRGB,
    WHITEPOINT_NAME_sRGB,
    MATRIX_sRGB_TO_XYZ,
    MATRIX_XYZ_TO_sRGB,
    eotf_inverse_sRGB,
    eotf_sRGB,
)
RGB_COLOURSPACE_sRGB.__doc__ = """
*sRGB* colourspace.

References
----------
:cite:`InternationalElectrotechnicalCommission1999a`,
:cite:`InternationalTelecommunicationUnion2015i`
"""
