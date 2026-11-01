"""
Ekta Space PS 5 Colourspace
===========================

Define the *Ekta Space PS 5* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_EKTA_SPACE_PS_5`.

References
----------
-   :cite:`Lindbloom2014a` : Lindbloom, B. (2014). RGB Working Space
    Information. Retrieved April 11, 2014, from
    http://www.brucelindbloom.com/WorkingSpaceInfo.html
-   :cite:`Holmesa` : Holmes, J. (n.d.). Ekta Space PS 5.
    https://www.josephholmes.com/userfiles/Ekta_Space_PS5_JHolmes.zip
"""

from __future__ import annotations

import typing
from functools import partial

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat

from colour.models.rgb import RGB_Colourspace, gamma_function, normalised_primary_matrix

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_EKTA_SPACE_PS_5",
    "WHITEPOINT_NAME_EKTA_SPACE_PS_5_V",
    "CCS_WHITEPOINT_EKTA_SPACE_PS_5",
    "MATRIX_EKTA_SPACE_PS_5_TO_XYZ",
    "MATRIX_XYZ_TO_EKTA_SPACE_PS_5",
    "RGB_COLOURSPACE_EKTA_SPACE_PS_5",
]

PRIMARIES_EKTA_SPACE_PS_5: NDArrayFloat = np.array(
    [
        [0.6950, 0.3050],
        [0.2600, 0.7000],
        [0.1100, 0.0050],
    ]
)
"""*Ekta Space PS 5* colourspace primaries."""

WHITEPOINT_NAME_EKTA_SPACE_PS_5_V: str = "D50"
"""*Ekta Space PS 5* colourspace whitepoint name."""

CCS_WHITEPOINT_EKTA_SPACE_PS_5: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_EKTA_SPACE_PS_5_V]
"""*Ekta Space PS 5* colourspace whitepoint chromaticity coordinates."""

MATRIX_EKTA_SPACE_PS_5_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_EKTA_SPACE_PS_5, CCS_WHITEPOINT_EKTA_SPACE_PS_5
)
"""*Ekta Space PS 5* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_EKTA_SPACE_PS_5: NDArrayFloat = np.linalg.inv(
    MATRIX_EKTA_SPACE_PS_5_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *Ekta Space PS 5* colourspace matrix."""

RGB_COLOURSPACE_EKTA_SPACE_PS_5: RGB_Colourspace = RGB_Colourspace(
    "Ekta Space PS 5",
    PRIMARIES_EKTA_SPACE_PS_5,
    CCS_WHITEPOINT_EKTA_SPACE_PS_5,
    WHITEPOINT_NAME_EKTA_SPACE_PS_5_V,
    MATRIX_EKTA_SPACE_PS_5_TO_XYZ,
    MATRIX_XYZ_TO_EKTA_SPACE_PS_5,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_EKTA_SPACE_PS_5.__doc__ = """
*Ekta Space PS 5* colourspace.

References
----------
:cite:`Holmesa`, :cite:`Lindbloom2014a`
"""
