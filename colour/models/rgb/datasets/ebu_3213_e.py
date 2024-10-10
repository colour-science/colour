"""
EBU Tech. 3213-E Colourspace
============================

Define the *EBU Tech. 3213-E* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_EBU_3213_E`.

References
----------
-   :cite:`EuropeanBroadcastingUnion1975` : European Broadcasting Union.
    (1975). EBU Tech 3213 - EBU Standard for Chromaticity Tolerances for Studio
    Monitors. https://tech.ebu.ch/docs/tech/tech3213.pdf
"""

from __future__ import annotations

import numpy as np

from colour.hints import NDArrayFloat
from colour.models.rgb import (
    RGB_Colourspace,
    linear_function,
    normalised_primary_matrix,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_EBU_3213_E",
    "WHITEPOINT_NAME_EBU_3213_E",
    "CCS_WHITEPOINT_EBU_3213_E",
    "MATRIX_EBU_3213_E_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_EBU_3213_E_RGB",
    "RGB_COLOURSPACE_EBU_3213_E",
]

PRIMARIES_EBU_3213_E: NDArrayFloat = np.array(
    [
        [0.64, 0.33],
        [0.29, 0.60],
        [0.15, 0.06],
    ]
)
"""*EBU Tech. 3213-E* colourspace primaries."""

WHITEPOINT_NAME_EBU_3213_E: str = "D65"
"""*EBU Tech. 3213-E* colourspace whitepoint name."""

CCS_WHITEPOINT_EBU_3213_E: NDArrayFloat = np.array([0.313, 0.329])
"""*EBU Tech. 3213-E* colourspace whitepoint chromaticity coordinates."""

MATRIX_EBU_3213_E_RGB_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_EBU_3213_E, CCS_WHITEPOINT_EBU_3213_E
)
"""*EBU Tech. 3213-E* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_EBU_3213_E_RGB: NDArrayFloat = np.linalg.inv(MATRIX_EBU_3213_E_RGB_TO_XYZ)
"""*CIE XYZ* tristimulus values to *EBU Tech. 3213-E* colourspace matrix."""

RGB_COLOURSPACE_EBU_3213_E: RGB_Colourspace = RGB_Colourspace(
    "EBU Tech. 3213-E",
    PRIMARIES_EBU_3213_E,
    CCS_WHITEPOINT_EBU_3213_E,
    WHITEPOINT_NAME_EBU_3213_E,
    MATRIX_EBU_3213_E_RGB_TO_XYZ,
    MATRIX_XYZ_TO_EBU_3213_E_RGB,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_EBU_3213_E.__doc__ = """
*EBU Tech. 3213-E* colourspace.

References
----------
:cite:`EuropeanBroadcastingUnion1975`
"""
