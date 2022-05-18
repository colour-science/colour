"""
EBU Tech. 3213-E Primaries and White Point
==========================================

Defines primaries and white point chromaticity coordinates from EBU Tech.
3213-E (:cite:`EuropeanBroadcastingUnion3213E`).

References
----------
-   :cite:`EuropeanBroadcastingUnion3213E` : European Broadcasting Union
    Tech. 3213-E (1975), Standard for Chromaticity Tolerances For Studio
    Monitors.
    https://tech.ebu.ch/docs/tech/tech3213.pdf
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
    "PRIMARIES_EBU_3213_E",
    "WHITEPOINT_NAME_EBU_3213_E",
    "CCS_WHITEPOINT_EBU_3213_E",
    "MATRIX_EBU_3213_E_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_EBU_3213_E_RGB",
]

PRIMARIES_EBU_3213_E: NDArray = np.array(
    [
        [0.64, 0.33],
        [0.29, 0.60],
        [0.15, 0.06],
    ]
)
"""Colourspace primaries for EBU Tech. 3213-E, as defined in
:cite:`EuropeanBroadcastingUnion3213E`."""

WHITEPOINT_NAME_EBU_3213_E: str = "D65"
"""Whitepoint name for EBU Tech. 3213-E, as defined in
:cite:`EuropeanBroadcastingUnion3213E`."""

CCS_WHITEPOINT_EBU_3213_E: NDArray = np.array([0.313, 0.329])
"""Whitepoint chromaticity coordinates for EBU Tech. 3213-E, as defined in
:cite:`EuropeanBroadcastingUnion3213E`."""

MATRIX_EBU_3213_E_RGB_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_EBU_3213_E, CCS_WHITEPOINT_EBU_3213_E
)
"""EBU Tech. 3213-E colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_EBU_3213_E_RGB: NDArray = np.linalg.inv(
    MATRIX_EBU_3213_E_RGB_TO_XYZ
)
"""*CIE XYZ* tristimulus values to EBU Tech. 3213-E colourspace matrix."""
