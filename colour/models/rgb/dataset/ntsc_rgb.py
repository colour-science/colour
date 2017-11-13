#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NTSC RGB Colourspace
====================

Defines the *NTSC RGB* colourspace:

-   :attr:`NTSC_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (1998). Recommendation
        ITU-R BT.470-6 - CONVENTIONAL TELEVISION SYSTEMS (pp. 1–36).
        Retrieved from http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.470-6-199811-S!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

from colour.models.rgb import RGB_Colourspace
from colour.models.rgb.dataset.itur_bt_470 import (
    BT470_525_PRIMARIES, BT470_525_WHITEPOINT, BT470_525_ILLUMINANT,
    BT470_525_TO_XYZ_MATRIX, XYZ_TO_BT470_525_MATRIX, BT470_525_COLOURSPACE)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'NTSC_RGB_PRIMARIES', 'NTSC_RGB_ILLUMINANT', 'NTSC_RGB_WHITEPOINT',
    'NTSC_RGB_TO_XYZ_MATRIX', 'XYZ_TO_NTSC_RGB_MATRIX', 'NTSC_RGB_COLOURSPACE'
]

NTSC_RGB_PRIMARIES = BT470_525_PRIMARIES
"""
*NTSC RGB* colourspace primaries.

NTSC_RGB_PRIMARIES : ndarray, (3, 2)
"""

NTSC_RGB_ILLUMINANT = BT470_525_ILLUMINANT
"""
*NTSC RGB* colourspace whitepoint name as illuminant.

NTSC_RGB_ILLUMINANT : unicode
"""

NTSC_RGB_WHITEPOINT = BT470_525_WHITEPOINT
"""
*NTSC RGB* colourspace whitepoint.

NTSC_RGB_WHITEPOINT : ndarray
"""

NTSC_RGB_TO_XYZ_MATRIX = BT470_525_TO_XYZ_MATRIX
"""
*NTSC RGB* colourspace to *CIE XYZ* tristimulus values matrix.

NTSC_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_NTSC_RGB_MATRIX = XYZ_TO_BT470_525_MATRIX
"""
*CIE XYZ* tristimulus values to *NTSC RGB* colourspace matrix.

XYZ_TO_NTSC_RGB_MATRIX : array_like, (3, 3)
"""

NTSC_RGB_COLOURSPACE = RGB_Colourspace(
    'NTSC RGB',
    NTSC_RGB_PRIMARIES,
    NTSC_RGB_WHITEPOINT,
    NTSC_RGB_ILLUMINANT,
    NTSC_RGB_TO_XYZ_MATRIX,
    XYZ_TO_NTSC_RGB_MATRIX,
    BT470_525_COLOURSPACE.encoding_cctf,
    BT470_525_COLOURSPACE.decoding_cctf)  # yapf: disable
"""
*NTSC RGB* colourspace.

NTSC_RGB_COLOURSPACE : RGB_Colourspace
"""
