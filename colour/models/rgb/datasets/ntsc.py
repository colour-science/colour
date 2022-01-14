# -*- coding: utf-8 -*-
"""
NTSC Colourspaces
=================

Defines the *NTSC* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_NTSC1953`.
-   :attr:`colour.models.RGB_COLOURSPACE_NTSC1987`.

References
----------
-   :cite:`InternationalTelecommunicationUnion1998a` : International
    Telecommunication Union. (1998). Recommendation ITU-R BT.470-6 -
    CONVENTIONAL TELEVISION SYSTEMS (pp. 1-36).
    http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.470-6-199811-S!!PDF-E.pdf
-   :cite:`SocietyofMotionPictureandTelevisionEngineers2004a` : Society of
    Motion Picture and Television Engineers. (2004). RP 145:2004: SMPTE C Color
    Monitor Colorimetry. In RP 145:2004: Vol. RP 145:200. The Society of Motion
    Picture and Television Engineers. doi:10.5594/S9781614821649
"""

from __future__ import annotations

from colour.hints import NDArray
from colour.models.rgb import RGB_Colourspace
from colour.models.rgb.datasets.itur_bt_470 import (
    PRIMARIES_BT470_525,
    CCS_WHITEPOINT_BT470_525,
    WHITEPOINT_NAME_BT470_525,
    MATRIX_BT470_525_TO_XYZ,
    MATRIX_XYZ_TO_BT470_525,
    RGB_COLOURSPACE_BT470_525,
)
from colour.models.rgb.datasets.smpte_c import (
    PRIMARIES_SMPTE_C,
    WHITEPOINT_NAME_SMPTE_C,
    CCS_WHITEPOINT_SMPTE_C,
    MATRIX_SMPTE_C_TO_XYZ,
    MATRIX_XYZ_TO_SMPTE_C,
    RGB_COLOURSPACE_SMPTE_C,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_NTSC1953',
    'WHITEPOINT_NAME_NTSC1953',
    'CCS_WHITEPOINT_NTSC1953',
    'MATRIX_NTSC1953_TO_XYZ',
    'MATRIX_XYZ_TO_NTSC1953',
    'RGB_COLOURSPACE_NTSC1953',
    'PRIMARIES_NTSC1987',
    'WHITEPOINT_NAME_NTSC1987',
    'CCS_WHITEPOINT_NTSC1987',
    'MATRIX_NTSC1987_TO_XYZ',
    'MATRIX_XYZ_TO_NTSC1987',
    'RGB_COLOURSPACE_NTSC1987',
]

PRIMARIES_NTSC1953: NDArray = PRIMARIES_BT470_525
"""
*NTSC (1953)* colourspace primaries.
"""

WHITEPOINT_NAME_NTSC1953: str = WHITEPOINT_NAME_BT470_525
"""
*NTSC (1953)* colourspace whitepoint name.
"""

CCS_WHITEPOINT_NTSC1953: NDArray = CCS_WHITEPOINT_BT470_525
"""
*NTSC (1953)* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_NTSC1953_TO_XYZ: NDArray = MATRIX_BT470_525_TO_XYZ
"""
*NTSC (1953)* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_NTSC1953: NDArray = MATRIX_XYZ_TO_BT470_525
"""
*CIE XYZ* tristimulus values to *NTSC (1953)* colourspace matrix.
"""

RGB_COLOURSPACE_NTSC1953: RGB_Colourspace = RGB_Colourspace(
    'NTSC (1953)',
    PRIMARIES_NTSC1953,
    CCS_WHITEPOINT_NTSC1953,
    WHITEPOINT_NAME_NTSC1953,
    MATRIX_NTSC1953_TO_XYZ,
    MATRIX_XYZ_TO_NTSC1953,
    RGB_COLOURSPACE_BT470_525.cctf_encoding,
    RGB_COLOURSPACE_BT470_525.cctf_decoding,
)
RGB_COLOURSPACE_NTSC1953.__doc__ = """
*NTSC (1953)* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion1998a`
"""

PRIMARIES_NTSC1987: NDArray = PRIMARIES_SMPTE_C
"""
*NTSC (1987)* colourspace primaries.
"""

WHITEPOINT_NAME_NTSC1987: str = WHITEPOINT_NAME_SMPTE_C
"""
*NTSC (1987)* colourspace whitepoint name.
"""

CCS_WHITEPOINT_NTSC1987: NDArray = CCS_WHITEPOINT_SMPTE_C
"""
*NTSC (1987)* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_NTSC1987_TO_XYZ: NDArray = MATRIX_SMPTE_C_TO_XYZ
"""
*NTSC (1987)* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_NTSC1987: NDArray = MATRIX_XYZ_TO_SMPTE_C
"""
*CIE XYZ* tristimulus values to *NTSC (1987)* colourspace matrix.
"""

RGB_COLOURSPACE_NTSC1987: RGB_Colourspace = RGB_Colourspace(
    'NTSC (1987)',
    PRIMARIES_NTSC1987,
    CCS_WHITEPOINT_NTSC1987,
    WHITEPOINT_NAME_NTSC1987,
    MATRIX_NTSC1987_TO_XYZ,
    MATRIX_XYZ_TO_NTSC1987,
    RGB_COLOURSPACE_SMPTE_C.cctf_encoding,
    RGB_COLOURSPACE_SMPTE_C.cctf_decoding,
)
RGB_COLOURSPACE_NTSC1987.__doc__ = """
*NTSC (1987)* colourspace.

References
----------
:cite:`SocietyofMotionPictureandTelevisionEngineers2004a`
"""
