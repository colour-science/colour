"""
Best RGB Colourspace
====================

Defines the *Best RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_BEST_RGB`.

References
----------
-   :cite:`HutchColord` : HutchColor. (n.d.). BestRGB (4 K).
    http://www.hutchcolor.com/profiles/BestRGB.zip
"""

from __future__ import annotations

from functools import partial

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArrayFloat
from colour.models.rgb import (
    RGB_Colourspace,
    gamma_function,
    normalised_primary_matrix,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_BEST_RGB",
    "WHITEPOINT_NAME_BEST_RGB",
    "CCS_WHITEPOINT_BEST_RGB",
    "MATRIX_BEST_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_BEST_RGB",
    "RGB_COLOURSPACE_BEST_RGB",
]

PRIMARIES_BEST_RGB: NDArrayFloat = np.array(
    [
        [0.735191637630662, 0.264808362369338],
        [0.215336134453781, 0.774159663865546],
        [0.130122950819672, 0.034836065573770],
    ]
)
"""*Best RGB* colourspace primaries."""

WHITEPOINT_NAME_BEST_RGB: str = "D50"
"""*Best RGB* colourspace whitepoint name."""

CCS_WHITEPOINT_BEST_RGB: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_BEST_RGB]
"""*Best RGB* colourspace whitepoint chromaticity coordinates."""

MATRIX_BEST_RGB_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_BEST_RGB, CCS_WHITEPOINT_BEST_RGB
)
"""*Best RGB* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_BEST_RGB: NDArrayFloat = np.linalg.inv(MATRIX_BEST_RGB_TO_XYZ)
"""*CIE XYZ* tristimulus values to *Best RGB* colourspace matrix."""

RGB_COLOURSPACE_BEST_RGB: RGB_Colourspace = RGB_Colourspace(
    "Best RGB",
    PRIMARIES_BEST_RGB,
    CCS_WHITEPOINT_BEST_RGB,
    WHITEPOINT_NAME_BEST_RGB,
    MATRIX_BEST_RGB_TO_XYZ,
    MATRIX_XYZ_TO_BEST_RGB,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_BEST_RGB.__doc__ = """
*Best RGB* colourspace.

References
----------
:cite:`HutchColord`
"""
