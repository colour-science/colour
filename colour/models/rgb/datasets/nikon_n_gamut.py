# -*- coding: utf-8 -*-
"""
Nikon N-Gamut Colourspace
=========================

Defines the *Nikon N-Gamut* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_N_GAMUT`.

References
----------
-   :cite:`Nikon2018` : Nikon. (2018). N-Log Specification Document - Version
    1.0.0 (pp. 1-5). Retrieved September 9, 2019, from
    http://download.nikonimglib.com/archive3/hDCmK00m9JDI03RPruD74xpoU905/\
N-Log_Specification_(En)01.pdf
"""

from __future__ import annotations

from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    log_encoding_NLog,
    log_decoding_NLog,
)
from colour.models.rgb.datasets.itur_bt_2020 import (
    PRIMARIES_BT2020,
    WHITEPOINT_NAME_BT2020,
    CCS_WHITEPOINT_BT2020,
    MATRIX_BT2020_TO_XYZ,
    MATRIX_XYZ_TO_BT2020,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_N_GAMUT',
    'WHITEPOINT_NAME_N_GAMUT',
    'CCS_WHITEPOINT_N_GAMUT',
    'MATRIX_N_GAMUT_TO_XYZ',
    'MATRIX_XYZ_TO_N_GAMUT',
    'RGB_COLOURSPACE_N_GAMUT',
]

PRIMARIES_N_GAMUT: NDArray = PRIMARIES_BT2020
"""
*Nikon N-Gamut* colourspace primaries.

Notes
-----
The *Nikon N-Gamut* colourspace gamut is same as the "ITU-R BT.2020" wide
colour gamut.
"""

WHITEPOINT_NAME_N_GAMUT: str = WHITEPOINT_NAME_BT2020
"""
*Nikon N-Gamut* colourspace whitepoint name.
"""

CCS_WHITEPOINT_N_GAMUT: NDArray = CCS_WHITEPOINT_BT2020
"""
*Nikon N-Gamut* colourspace whitepoint.
"""

MATRIX_N_GAMUT_TO_XYZ: NDArray = MATRIX_BT2020_TO_XYZ
"""
*Nikon N-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_N_GAMUT: NDArray = MATRIX_XYZ_TO_BT2020
"""
*CIE XYZ* tristimulus values to *Nikon N-Gamut* colourspace matrix.
"""

RGB_COLOURSPACE_N_GAMUT: RGB_Colourspace = RGB_Colourspace(
    'N-Gamut',
    PRIMARIES_N_GAMUT,
    CCS_WHITEPOINT_N_GAMUT,
    WHITEPOINT_NAME_N_GAMUT,
    MATRIX_N_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_N_GAMUT,
    log_encoding_NLog,
    log_decoding_NLog,
)
RGB_COLOURSPACE_N_GAMUT.__doc__ = """
*Nikon N-Gamut* colourspace.

References
----------
:cite:`Nikon2018`
"""
