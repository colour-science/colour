"""
ITU-T H.273 video primaries and whitepoints
===========================================

Contains several primaries and whitepoints that are defined in ITU-T H.273
(:cite:`ITU2021`) but don't belong in another specification or standard.

References
----------
-   :cite:`ITU2021` : International Telecommunication Union. (2021). Recommendation
    ITU-T H.273 - Coding-independent code points for video signal type identification.
    https://www.itu.int/rec/T-REC-H.273-202107-I/en
"""

from __future__ import annotations

import numpy as np

from colour.hints import NDArray
from colour.models.rgb import normalised_primary_matrix

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_FILM_C",
    "WHITEPOINT_NAME_FILM_C",
    "CCS_WHITEPOINT_FILM_C",
    "MATRIX_FILM_C_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_FILM_C_RGB",
    "PRIMARIES_ITUT_H273_22",
    "WHITEPOINT_NAME_ITUT_H273_22",
    "CCS_WHITEPOINT_ITUT_H273_22",
    "MATRIX_ITUT_H273_22_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_ITUT_H273_22_RGB",
]

PRIMARIES_FILM_C: NDArray = np.array(
    [
        [0.681, 0.319],
        [0.243, 0.692],
        [0.145, 0.049],
    ]
)
"""Colourspace primaries for "colour filters using illuminant C", as defined in
:cite:`ITU2021`."""

WHITEPOINT_NAME_FILM_C: str = "C"
"""Whitepoint name for "colour filters using illuminant C", as defined in
:cite:`ITU2021`."""

# Note: ITU-T H.273 defines white point C as [0.310, 0.316], while colour
# has a slightly higher precision.
CCS_WHITEPOINT_FILM_C: NDArray = np.array([0.310, 0.316])
"""Whitepoint chromaticity coordinates for "colour filters using illuminant C", as
defined in :cite:`ITU2021`."""

MATRIX_FILM_C_RGB_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_FILM_C, CCS_WHITEPOINT_FILM_C
)
"""'Colour filters using Illuminant C' colourspace to *CIE XYZ* tristimulus values
matrix."""

MATRIX_XYZ_TO_FILM_C_RGB: NDArray = np.linalg.inv(MATRIX_FILM_C_RGB_TO_XYZ)
"""*CIE XYZ* tristimulus values to 'Colour filters using Illuminant C' colourspace
matrix."""

PRIMARIES_ITUT_H273_22: NDArray = np.array(
    [
        [0.630, 0.340],
        [0.295, 0.605],
        [0.155, 0.077],
    ]
)
"""Colourspace primaries for ColourPrimaries number 22 defined in :cite:`ITU2021`."""

WHITEPOINT_NAME_ITUT_H273_22: str = "D65"
"""Whitepoint name for ColourPrimaries number 22 defined in :cite:`ITU2021`."""

CCS_WHITEPOINT_ITUT_H273_22: NDArray = np.array([0.3127, 0.3290])
"""Whitepoint chromaticity coordinates for ColourPrimaries number 22, as defined in
:cite:`ITU2021`."""

MATRIX_ITUT_H273_22_RGB_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_ITUT_H273_22, CCS_WHITEPOINT_ITUT_H273_22
)
"""ITU-T H.273 ColourPrimaries number 22 to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_ITUT_H273_22_RGB: NDArray = np.linalg.inv(
    MATRIX_ITUT_H273_22_RGB_TO_XYZ
)
"""*CIE XYZ* tristimulus values to ITU-T H.273 ColourPrimaries number 22 matrix."""
