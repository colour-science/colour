# -*- coding: utf-8 -*-
"""
NTSC Colourspaces
=================

Defines the *NTSC* colourspaces:

-   :attr:`colour.models.NTSC_1953_COLOURSPACE`.
-   :attr:`colour.models.NTSC_1987_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`InternationalTelecommunicationUnion1998a` : International
    Telecommunication Union. (1998). Recommendation ITU-R BT.470-6 -
    CONVENTIONAL TELEVISION SYSTEMS. Retrieved from http://www.itu.int/\
dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf
-   :cite:`SocietyofMotionPictureandTelevisionEngineers2004a` : Society of
Motion Picture and Television Engineers. (2004). RP 145:2004: SMPTE C Color
Monitor Colorimetry. RP 145:2004 (Vol. RP 145:200). The Society of Motion
Picture and Television Engineers. doi:10.5594/S9781614821649
"""

from __future__ import division, unicode_literals

from colour.models.rgb import RGB_Colourspace
from colour.models.rgb.datasets.itur_bt_470 import (
    BT470_525_PRIMARIES, BT470_525_WHITEPOINT, BT470_525_WHITEPOINT_NAME,
    BT470_525_TO_XYZ_MATRIX, XYZ_TO_BT470_525_MATRIX, BT470_525_COLOURSPACE)
from colour.models.rgb.datasets.smpte_c import (
    SMPTE_C_PRIMARIES, SMPTE_C_WHITEPOINT_NAME, SMPTE_C_WHITEPOINT,
    SMPTE_C_TO_XYZ_MATRIX, XYZ_TO_SMPTE_C_MATRIX, SMPTE_C_COLOURSPACE)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'NTSC_1953_PRIMARIES', 'NTSC_1953_WHITEPOINT_NAME', 'NTSC_1953_WHITEPOINT',
    'NTSC_1953_TO_XYZ_MATRIX', 'XYZ_TO_NTSC_1953_MATRIX',
    'NTSC_1953_COLOURSPACE', 'NTSC_1987_PRIMARIES',
    'NTSC_1987_WHITEPOINT_NAME', 'NTSC_1987_WHITEPOINT',
    'NTSC_1987_TO_XYZ_MATRIX', 'XYZ_TO_NTSC_1987_MATRIX',
    'NTSC_1987_COLOURSPACE'
]

NTSC_1953_PRIMARIES = BT470_525_PRIMARIES
"""
*NTSC (1953)* colourspace primaries.

NTSC_1953_PRIMARIES : ndarray, (3, 2)
"""

NTSC_1953_WHITEPOINT_NAME = BT470_525_WHITEPOINT_NAME
"""
*NTSC (1953)* colourspace whitepoint name.

NTSC_1953_WHITEPOINT_NAME : unicode
"""

NTSC_1953_WHITEPOINT = BT470_525_WHITEPOINT
"""
*NTSC (1953)* colourspace whitepoint.

NTSC_1953_WHITEPOINT : ndarray
"""

NTSC_1953_TO_XYZ_MATRIX = BT470_525_TO_XYZ_MATRIX
"""
*NTSC (1953)* colourspace to *CIE XYZ* tristimulus values matrix.

NTSC_1953_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_NTSC_1953_MATRIX = XYZ_TO_BT470_525_MATRIX
"""
*CIE XYZ* tristimulus values to *NTSC (1953)* colourspace matrix.

XYZ_TO_NTSC_1953_MATRIX : array_like, (3, 3)
"""

NTSC_1953_COLOURSPACE = RGB_Colourspace(
    'NTSC (1953)',
    NTSC_1953_PRIMARIES,
    NTSC_1953_WHITEPOINT,
    NTSC_1953_WHITEPOINT_NAME,
    NTSC_1953_TO_XYZ_MATRIX,
    XYZ_TO_NTSC_1953_MATRIX,
    BT470_525_COLOURSPACE.cctf_encoding,
    BT470_525_COLOURSPACE.cctf_decoding,
)
NTSC_1953_COLOURSPACE.__doc__ = """
*NTSC (1953)* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion1998a`

NTSC_1953_COLOURSPACE : RGB_Colourspace
"""

NTSC_1987_PRIMARIES = SMPTE_C_PRIMARIES
"""
*NTSC (1987)* colourspace primaries.

NTSC_1987_PRIMARIES : ndarray, (3, 2)
"""

NTSC_1987_WHITEPOINT_NAME = SMPTE_C_WHITEPOINT_NAME
"""
*NTSC (1987)* colourspace whitepoint name.

NTSC_1987_WHITEPOINT_NAME : unicode
"""

NTSC_1987_WHITEPOINT = SMPTE_C_WHITEPOINT
"""
*NTSC (1987)* colourspace whitepoint.

NTSC_1987_WHITEPOINT : ndarray
"""

NTSC_1987_TO_XYZ_MATRIX = SMPTE_C_TO_XYZ_MATRIX
"""
*NTSC (1987)* colourspace to *CIE XYZ* tristimulus values matrix.

NTSC_1987_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_NTSC_1987_MATRIX = XYZ_TO_SMPTE_C_MATRIX
"""
*CIE XYZ* tristimulus values to *NTSC (1987)* colourspace matrix.

XYZ_TO_NTSC_1987_MATRIX : array_like, (3, 3)
"""

NTSC_1987_COLOURSPACE = RGB_Colourspace(
    'NTSC (1987)',
    NTSC_1987_PRIMARIES,
    NTSC_1987_WHITEPOINT,
    NTSC_1987_WHITEPOINT_NAME,
    NTSC_1987_TO_XYZ_MATRIX,
    XYZ_TO_NTSC_1987_MATRIX,
    SMPTE_C_COLOURSPACE.cctf_encoding,
    SMPTE_C_COLOURSPACE.cctf_decoding,
)
NTSC_1987_COLOURSPACE.__doc__ = """
*NTSC (1987)* colourspace.

References
----------
:cite:`SocietyofMotionPictureandTelevisionEngineers2004a`

NTSC_1987_COLOURSPACE : RGB_Colourspace
"""
