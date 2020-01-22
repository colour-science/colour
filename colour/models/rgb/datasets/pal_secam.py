# -*- coding: utf-8 -*-
"""
Pal/Secam Colourspace
=====================

Defines the *Pal/Secam* colourspace:

-   :attr:`colour.models.PAL_SECAM_COLOURSPACE`.

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
"""

from __future__ import division, unicode_literals

from colour.models.rgb import RGB_Colourspace
from colour.models.rgb.datasets.itur_bt_470 import (
    BT470_625_PRIMARIES, BT470_625_WHITEPOINT, BT470_625_WHITEPOINT_NAME,
    BT470_625_TO_XYZ_MATRIX, XYZ_TO_BT470_625_MATRIX, BT470_625_COLOURSPACE)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PAL_SECAM_PRIMARIES', 'PAL_SECAM_WHITEPOINT_NAME', 'PAL_SECAM_WHITEPOINT',
    'PAL_SECAM_TO_XYZ_MATRIX', 'XYZ_TO_PAL_SECAM_MATRIX',
    'PAL_SECAM_COLOURSPACE'
]

PAL_SECAM_PRIMARIES = BT470_625_PRIMARIES
"""
*Pal/Secam* colourspace primaries.

PAL_SECAM_PRIMARIES : ndarray, (3, 2)
"""

PAL_SECAM_WHITEPOINT_NAME = BT470_625_WHITEPOINT_NAME
"""
*Pal/Secam* colourspace whitepoint name.

PAL_SECAM_WHITEPOINT_NAME : unicode
"""

PAL_SECAM_WHITEPOINT = BT470_625_WHITEPOINT
"""
*Pal/Secam* colourspace whitepoint.

PAL_SECAM_WHITEPOINT : ndarray
"""

PAL_SECAM_TO_XYZ_MATRIX = BT470_625_TO_XYZ_MATRIX
"""
*Pal/Secam* colourspace to *CIE XYZ* tristimulus values matrix.

PAL_SECAM_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_PAL_SECAM_MATRIX = XYZ_TO_BT470_625_MATRIX
"""
*CIE XYZ* tristimulus values to *Pal/Secam* colourspace matrix.

XYZ_TO_PAL_SECAM_MATRIX : array_like, (3, 3)
"""

PAL_SECAM_COLOURSPACE = RGB_Colourspace(
    'Pal/Secam',
    PAL_SECAM_PRIMARIES,
    PAL_SECAM_WHITEPOINT,
    PAL_SECAM_WHITEPOINT_NAME,
    PAL_SECAM_TO_XYZ_MATRIX,
    XYZ_TO_PAL_SECAM_MATRIX,
    BT470_625_COLOURSPACE.cctf_encoding,
    BT470_625_COLOURSPACE.cctf_decoding,
)
PAL_SECAM_COLOURSPACE.__doc__ = """
*Pal/Secam* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion1998a`

PAL_SECAM_COLOURSPACE : RGB_Colourspace
"""
