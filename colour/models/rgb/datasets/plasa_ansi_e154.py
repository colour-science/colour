"""
PLASA ANSI E1.54 Colourspace
============================

Defines the *PLASA ANSI E1.54* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_PLASA_ANSI_E154`.

References
----------
-   :cite:`Wood2014` : Wood, M. (2014). Making the same color twice - A
    proposed PLASA standard for color communication. Retrieved August 13, 2023,
    from https://www.mikewoodconsulting.com/articles/\
Protocol%20Fall%202014%20-%20Color%20Communication.pdf
-   :cite:`PLASANorthAmerica2015` : PLASA North America. (2015). ANSI E1.54 -
    2015 - PLASA Standard for Color Communication in Entertainment Lighting.
    https://webstore.ansi.org/preview-pages/ESTA/preview_ANSI+E1.54-2015.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArrayFloat
from colour.models.rgb import (
    RGB_Colourspace,
    linear_function,
    normalised_primary_matrix,
)
from colour.models.rgb.datasets import RGB_COLOURSPACE_RIMM_RGB

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_PLASA_ANSI_E154",
    "WHITEPOINT_NAME_PLASA_ANSI_E154",
    "CCS_WHITEPOINT_PLASA_ANSI_E154",
    "MATRIX_PLASA_ANSI_E154_TO_XYZ",
    "MATRIX_XYZ_TO_PLASA_ANSI_E154",
    "RGB_COLOURSPACE_PLASA_ANSI_E154",
]

PRIMARIES_PLASA_ANSI_E154: NDArrayFloat = RGB_COLOURSPACE_RIMM_RGB.primaries
"""*PLASA ANSI E1.54* colourspace primaries."""

WHITEPOINT_NAME_PLASA_ANSI_E154: str = "PLASA ANSI E1.54"
"""*PLASA ANSI E1.54* colourspace whitepoint name."""

CCS_WHITEPOINT_PLASA_ANSI_E154: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_PLASA_ANSI_E154]
"""*PLASA ANSI E1.54* colourspace whitepoint chromaticity coordinates."""

MATRIX_PLASA_ANSI_E154_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_PLASA_ANSI_E154, CCS_WHITEPOINT_PLASA_ANSI_E154
)
"""*PLASA ANSI E1.54* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_PLASA_ANSI_E154: NDArrayFloat = np.linalg.inv(
    MATRIX_PLASA_ANSI_E154_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *PLASA ANSI E1.54* colourspace matrix."""

RGB_COLOURSPACE_PLASA_ANSI_E154: RGB_Colourspace = RGB_Colourspace(
    "PLASA ANSI E1.54",
    PRIMARIES_PLASA_ANSI_E154,
    CCS_WHITEPOINT_PLASA_ANSI_E154,
    WHITEPOINT_NAME_PLASA_ANSI_E154,
    MATRIX_PLASA_ANSI_E154_TO_XYZ,
    MATRIX_XYZ_TO_PLASA_ANSI_E154,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_PLASA_ANSI_E154.__doc__ = """
*PLASA ANSI E1.54* colourspace.

Notes
-----
The `[0.4254, 0.4044]` whitepoint chromaticity coordinates are described by
:cite:`Wood2014` to be that of a "2° Planckian source at 3,200 K". However, we
can show that the chromaticity coordinates should be `[0.4234, 0.3990]`::

    sd = colour.sd_blackbody(3200)
    colour.XYZ_to_xy(
        colour.sd_to_XYZ(
            sd, colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        )
    ).round(4)

References
----------
:cite:`PLASANorthAmerica2015`, :cite:`Wood2014`
"""
