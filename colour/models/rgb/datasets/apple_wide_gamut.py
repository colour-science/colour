"""
Apple Wide Gamut Colourspace
===========================

Define the *Apple Wide Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_APPLE_WIDE_GAMUT`.

References
----------
-   :cite:`AppleInc.2025` : Apple Inc. (2025). Apple Log 2 White Paper.
    https://developer.apple.com/download/all/?q=Apple%20Log%202
"""

from __future__ import annotations

import typing

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat

from colour.models.rgb import (
    RGB_Colourspace,
    log_decoding_AppleLogProfile,
    log_encoding_AppleLogProfile,
    normalised_primary_matrix,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_APPLE_WIDE_GAMUT",
    "WHITEPOINT_NAME_APPLE_WIDE_GAMUT",
    "CCS_WHITEPOINT_APPLE_WIDE_GAMUT",
    "MATRIX_APPLE_WIDE_GAMUT_TO_XYZ",
    "MATRIX_XYZ_TO_APPLE_WIDE_GAMUT",
    "RGB_COLOURSPACE_APPLE_WIDE_GAMUT",
]

PRIMARIES_APPLE_WIDE_GAMUT: NDArrayFloat = np.array(
    [
        [0.725, 0.301],
        [0.221, 0.814],
        [0.068, -0.076],
    ]
)
"""*Apple Wide Gamut* colourspace primaries."""

WHITEPOINT_NAME_APPLE_WIDE_GAMUT: str = "D65"
"""*Apple Wide Gamut* colourspace whitepoint name."""

CCS_WHITEPOINT_APPLE_WIDE_GAMUT: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_APPLE_WIDE_GAMUT]
"""*Apple Wide Gamut* colourspace whitepoint chromaticity coordinates."""

MATRIX_APPLE_WIDE_GAMUT_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_APPLE_WIDE_GAMUT, CCS_WHITEPOINT_APPLE_WIDE_GAMUT
)
"""*Apple Wide Gamut* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_APPLE_WIDE_GAMUT: NDArrayFloat = np.linalg.inv(
    MATRIX_APPLE_WIDE_GAMUT_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *Apple Wide Gamut* colourspace matrix."""

RGB_COLOURSPACE_APPLE_WIDE_GAMUT: RGB_Colourspace = RGB_Colourspace(
    "Apple Wide Gamut",
    PRIMARIES_APPLE_WIDE_GAMUT,
    CCS_WHITEPOINT_APPLE_WIDE_GAMUT,
    WHITEPOINT_NAME_APPLE_WIDE_GAMUT,
    MATRIX_APPLE_WIDE_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_APPLE_WIDE_GAMUT,
    log_encoding_AppleLogProfile,
    log_decoding_AppleLogProfile,
)
RGB_COLOURSPACE_APPLE_WIDE_GAMUT.__doc__ = """
*Apple Wide Gamut* colourspace with the *Apple Log* transfer function,
forming the *Apple Log 2* encoding.

References
----------
:cite:`AppleInc.2025`
"""
