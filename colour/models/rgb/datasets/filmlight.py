"""
FilmLight Colourspaces
======================

Define the *FilmLight* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_FILMLIGHT_E_GAMUT`.
-   :attr:`colour.models.RGB_COLOURSPACE_FILMLIGHT_E_GAMUT_2`.

References
----------
-   :cite:`Siragusano2018a` : Siragusano, D. (2018). Private Discussion with
    Shaw, Nick.
-   :cite:`Siragusano2025` : Siragusano, D. (2025). Private discussion on
    colour-science Discord server. https://discord.com/channels/\
    1269935627386884106/1269935628808622102/1325770472058523668
"""

from __future__ import annotations

import typing

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat

from colour.models.rgb import (
    RGB_Colourspace,
    log_decoding_FilmLightTLog,
    log_encoding_FilmLightTLog,
    normalised_primary_matrix,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_FILMLIGHT_E_GAMUT",
    "WHITEPOINT_NAME_FILMLIGHT_E_GAMUT",
    "CCS_WHITEPOINT_FILMLIGHT_E_GAMUT",
    "MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ",
    "MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT",
    "RGB_COLOURSPACE_FILMLIGHT_E_GAMUT",
    "PRIMARIES_FILMLIGHT_E_GAMUT_2",
    "MATRIX_FILMLIGHT_E_GAMUT_2_TO_XYZ",
    "MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT_2",
    "RGB_COLOURSPACE_FILMLIGHT_E_GAMUT_2",
]

PRIMARIES_FILMLIGHT_E_GAMUT: NDArrayFloat = np.array(
    [
        [0.8000, 0.3177],
        [0.1800, 0.9000],
        [0.0650, -0.0805],
    ]
)
"""*FilmLight E-Gamut* colourspace primaries."""

WHITEPOINT_NAME_FILMLIGHT_E_GAMUT: str = "D65"
"""*FilmLight E-Gamut* colourspace whitepoint name."""

CCS_WHITEPOINT_FILMLIGHT_E_GAMUT: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_FILMLIGHT_E_GAMUT]
"""*FilmLight E-Gamut* colourspace whitepoint chromaticity coordinates."""

MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_FILMLIGHT_E_GAMUT, CCS_WHITEPOINT_FILMLIGHT_E_GAMUT
)
"""*FilmLight E-Gamut* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT: NDArrayFloat = np.linalg.inv(
    MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *FilmLight E-Gamut* colourspace matrix."""

RGB_COLOURSPACE_FILMLIGHT_E_GAMUT: RGB_Colourspace = RGB_Colourspace(
    "FilmLight E-Gamut",
    PRIMARIES_FILMLIGHT_E_GAMUT,
    CCS_WHITEPOINT_FILMLIGHT_E_GAMUT,
    WHITEPOINT_NAME_FILMLIGHT_E_GAMUT,
    MATRIX_FILMLIGHT_E_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT,
    log_encoding_FilmLightTLog,
    log_decoding_FilmLightTLog,
)
RGB_COLOURSPACE_FILMLIGHT_E_GAMUT.__doc__ = """
*FilmLight E-Gamut* colourspace.

References
----------
:cite:`Siragusano2018a`
"""

PRIMARIES_FILMLIGHT_E_GAMUT_2: NDArrayFloat = np.array(
    [
        [0.8300, 0.3100],
        [0.1500, 0.9500],
        [0.0650, -0.0805],
    ]
)
"""*FilmLight E-Gamut 2* colourspace primaries."""

MATRIX_FILMLIGHT_E_GAMUT_2_TO_XYZ: NDArrayFloat = np.array(
    [
        [0.736478, 0.130740, 0.083239],
        [0.275070, 0.828018, -0.103088],
        [-0.124225, -0.087160, 1.300443],
    ]
)
"""*FilmLight E-Gamut 2* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT_2: NDArrayFloat = np.linalg.inv(
    MATRIX_FILMLIGHT_E_GAMUT_2_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *FilmLight E-Gamut 2* colourspace matrix."""

RGB_COLOURSPACE_FILMLIGHT_E_GAMUT_2: RGB_Colourspace = RGB_Colourspace(
    "FilmLight E-Gamut 2",
    PRIMARIES_FILMLIGHT_E_GAMUT_2,
    CCS_WHITEPOINT_FILMLIGHT_E_GAMUT,
    WHITEPOINT_NAME_FILMLIGHT_E_GAMUT,
    MATRIX_FILMLIGHT_E_GAMUT_2_TO_XYZ,
    MATRIX_XYZ_TO_FILMLIGHT_E_GAMUT_2,
    log_encoding_FilmLightTLog,
    log_decoding_FilmLightTLog,
)
RGB_COLOURSPACE_FILMLIGHT_E_GAMUT_2.__doc__ = """
*FilmLight E-Gamut 2* colourspace.

References
----------
:cite:`Siragusano2025`
"""
