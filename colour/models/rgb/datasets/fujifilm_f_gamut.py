"""
Fujifilm F-Gamut Colourspace
============================

Define the *Fujifilm F-Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_F_GAMUT`.

References
----------
-   :cite:`Fujifilm2022a` : Fujifilm. (2022). F-Log Data Sheet Ver.1.1 (pp.
    1-4). https://dl.fujifilm-x.com/support/lut/F-Log_DataSheet_E_Ver.1.1.pdf
"""

from __future__ import annotations

from colour.hints import NDArrayFloat
from colour.models.rgb import (
    RGB_Colourspace,
    log_decoding_FLog,
    log_encoding_FLog,
)
from colour.models.rgb.datasets.itur_bt_2020 import (
    CCS_WHITEPOINT_BT2020,
    MATRIX_BT2020_TO_XYZ,
    MATRIX_XYZ_TO_BT2020,
    PRIMARIES_BT2020,
    WHITEPOINT_NAME_BT2020,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_F_GAMUT",
    "WHITEPOINT_NAME_F_GAMUT",
    "CCS_WHITEPOINT_F_GAMUT",
    "MATRIX_F_GAMUT_TO_XYZ",
    "MATRIX_XYZ_TO_F_GAMUT",
    "RGB_COLOURSPACE_F_GAMUT",
]

PRIMARIES_F_GAMUT: NDArrayFloat = PRIMARIES_BT2020
"""*Fujifilm F-Gamut* colourspace primaries."""

WHITEPOINT_NAME_F_GAMUT: str = WHITEPOINT_NAME_BT2020
"""*Fujifilm F-Gamut* colourspace whitepoint name."""

CCS_WHITEPOINT_F_GAMUT: NDArrayFloat = CCS_WHITEPOINT_BT2020
"""*Fujifilm F-Gamut* colourspace whitepoint chromaticity coordinates."""

MATRIX_F_GAMUT_TO_XYZ: NDArrayFloat = MATRIX_BT2020_TO_XYZ
"""*Fujifilm F-Gamut* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_F_GAMUT: NDArrayFloat = MATRIX_XYZ_TO_BT2020
"""*CIE XYZ* tristimulus values to *Fujifilm F-Gamut* colourspace matrix."""

RGB_COLOURSPACE_F_GAMUT: RGB_Colourspace = RGB_Colourspace(
    "F-Gamut",
    PRIMARIES_F_GAMUT,
    CCS_WHITEPOINT_F_GAMUT,
    WHITEPOINT_NAME_F_GAMUT,
    MATRIX_F_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_F_GAMUT,
    log_encoding_FLog,
    log_decoding_FLog,
)
RGB_COLOURSPACE_F_GAMUT.__doc__ = """
*Fujifilm F-Gamut* colourspace.

References
----------
:cite:`Fujifilm2022a`
"""
