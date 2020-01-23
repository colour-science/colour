# -*- coding: utf-8 -*-
"""
RIMM, ROMM and ERIMM Encodings
==============================

Defines the *RIMM, ROMM and ERIMM* encodings:

-   :attr:`colour.models.ROMM_RGB_COLOURSPACE`.
-   :attr:`colour.models.RIMM_RGB_COLOURSPACE`.
-   :attr:`colour.models.ERIMM_RGB_COLOURSPACE`.
-   :attr:`colour.models.PROPHOTO_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`ANSI2003a` : ANSI. (2003). Specification of ROMM RGB. Retrieved from
    http://www.color.org/ROMMRGB.pdf
-   :cite:`Spaulding2000b` : Spaulding, K. E., Woolfe, G. J., & Giorgianni,
    E. J. (2000). Reference Input/Output Medium Metric RGB Color Encodings
    (RIMM/ROMM RGB). Retrieved from
    http://www.photo-lovers.org/pdf/color/romm.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace, cctf_encoding_ROMMRGB, cctf_decoding_ROMMRGB,
    cctf_encoding_RIMMRGB, cctf_decoding_RIMMRGB, log_encoding_ERIMMRGB,
    log_decoding_ERIMMRGB, cctf_encoding_ProPhotoRGB,
    cctf_decoding_ProPhotoRGB)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'ROMM_RGB_PRIMARIES', 'ROMM_RGB_WHITEPOINT_NAME', 'ROMM_RGB_WHITEPOINT',
    'ROMM_RGB_TO_XYZ_MATRIX', 'XYZ_TO_ROMM_RGB_MATRIX', 'ROMM_RGB_COLOURSPACE',
    'RIMM_RGB_PRIMARIES', 'RIMM_RGB_WHITEPOINT_NAME', 'RIMM_RGB_WHITEPOINT',
    'RIMM_RGB_TO_XYZ_MATRIX', 'XYZ_TO_RIMM_RGB_MATRIX', 'RIMM_RGB_COLOURSPACE',
    'ERIMM_RGB_PRIMARIES', 'ERIMM_RGB_WHITEPOINT_NAME', 'ERIMM_RGB_WHITEPOINT',
    'ERIMM_RGB_TO_XYZ_MATRIX', 'XYZ_TO_ERIMM_RGB_MATRIX',
    'ERIMM_RGB_COLOURSPACE', 'PROPHOTO_RGB_PRIMARIES',
    'PROPHOTO_RGB_WHITEPOINT_NAME', 'PROPHOTO_RGB_WHITEPOINT',
    'PROPHOTO_RGB_TO_XYZ_MATRIX', 'XYZ_TO_PROPHOTO_RGB_MATRIX',
    'PROPHOTO_RGB_COLOURSPACE'
]

ROMM_RGB_PRIMARIES = np.array([
    [0.7347, 0.2653],
    [0.1596, 0.8404],
    [0.0366, 0.0001],
])
"""
*ROMM RGB* colourspace primaries.

ROMM_RGB_PRIMARIES : ndarray, (3, 2)
"""

ROMM_RGB_WHITEPOINT_NAME = 'D50'
"""
*ROMM RGB* colourspace whitepoint name.

ROMM_RGB_WHITEPOINT_NAME : unicode
"""

ROMM_RGB_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    ROMM_RGB_WHITEPOINT_NAME])
"""
*ROMM RGB* colourspace whitepoint.

ROMM_RGB_WHITEPOINT : ndarray
"""

ROMM_RGB_TO_XYZ_MATRIX = np.array([
    [0.7977, 0.1352, 0.0313],
    [0.2880, 0.7119, 0.0001],
    [0.0000, 0.0000, 0.8249],
])
"""
*ROMM RGB* colourspace to *CIE XYZ* tristimulus values matrix.

ROMM_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ROMM_RGB_MATRIX = np.array([
    [1.3460, -0.2556, -0.0511],
    [-0.5446, 1.5082, 0.0205],
    [0.0000, 0.0000, 1.2123],
])
"""
*CIE XYZ* tristimulus values to *ROMM RGB* colourspace matrix.

XYZ_TO_ROMM_RGB_MATRIX : array_like, (3, 3)
"""

ROMM_RGB_COLOURSPACE = RGB_Colourspace(
    'ROMM RGB',
    ROMM_RGB_PRIMARIES,
    ROMM_RGB_WHITEPOINT,
    ROMM_RGB_WHITEPOINT_NAME,
    ROMM_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ROMM_RGB_MATRIX,
    cctf_encoding_ROMMRGB,
    cctf_decoding_ROMMRGB,
)
ROMM_RGB_COLOURSPACE.__doc__ = """
*ROMM RGB* colourspace.

References
----------
:cite:`ANSI2003a`, :cite:`Spaulding2000b`

ROMM_RGB_COLOURSPACE : RGB_Colourspace
"""

RIMM_RGB_PRIMARIES = ROMM_RGB_PRIMARIES
"""
*RIMM RGB* colourspace primaries.

RIMM_RGB_PRIMARIES : ndarray, (3, 2)
"""

RIMM_RGB_WHITEPOINT_NAME = ROMM_RGB_WHITEPOINT_NAME
"""
*RIMM RGB* colourspace whitepoint name.

RIMM_RGB_WHITEPOINT_NAME : unicode
"""

RIMM_RGB_WHITEPOINT = ROMM_RGB_WHITEPOINT
"""
*RIMM RGB* colourspace whitepoint.

RIMM_RGB_WHITEPOINT : ndarray
"""

RIMM_RGB_TO_XYZ_MATRIX = ROMM_RGB_TO_XYZ_MATRIX
"""
*RIMM RGB* colourspace to *CIE XYZ* tristimulus values matrix.

RIMM_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_RIMM_RGB_MATRIX = XYZ_TO_ROMM_RGB_MATRIX
"""
*CIE XYZ* tristimulus values to *RIMM RGB* colourspace matrix.

XYZ_TO_RIMM_RGB_MATRIX : array_like, (3, 3)
"""

RIMM_RGB_COLOURSPACE = RGB_Colourspace(
    'RIMM RGB',
    RIMM_RGB_PRIMARIES,
    RIMM_RGB_WHITEPOINT,
    RIMM_RGB_WHITEPOINT_NAME,
    RIMM_RGB_TO_XYZ_MATRIX,
    XYZ_TO_RIMM_RGB_MATRIX,
    cctf_encoding_RIMMRGB,
    cctf_decoding_RIMMRGB,
)
RIMM_RGB_COLOURSPACE.__doc__ = """
*RIMM RGB* colourspace. In cases in which it is necessary to identify a
specific precision level, the notation *RIMM8 RGB*, *RIMM12 RGB* and
*RIMM16 RGB* is used.

References
----------
:cite:`Spaulding2000b`

RIMM_RGB_COLOURSPACE : RGB_Colourspace
"""

ERIMM_RGB_PRIMARIES = ROMM_RGB_PRIMARIES
"""
*ERIMM RGB* colourspace primaries.

ERIMM_RGB_PRIMARIES : ndarray, (3, 2)
"""

ERIMM_RGB_WHITEPOINT_NAME = ROMM_RGB_WHITEPOINT_NAME
"""
*ERIMM RGB* colourspace whitepoint name.

ERIMM_RGB_WHITEPOINT_NAME : unicode
"""

ERIMM_RGB_WHITEPOINT = ROMM_RGB_WHITEPOINT
"""
*ERIMM RGB* colourspace whitepoint.

ERIMM_RGB_WHITEPOINT : ndarray
"""

ERIMM_RGB_TO_XYZ_MATRIX = ROMM_RGB_TO_XYZ_MATRIX
"""
*ERIMM RGB* colourspace to *CIE XYZ* tristimulus values matrix.

ERIMM_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ERIMM_RGB_MATRIX = XYZ_TO_ROMM_RGB_MATRIX
"""
*CIE XYZ* tristimulus values to *ERIMM RGB* colourspace matrix.

XYZ_TO_ERIMM_RGB_MATRIX : array_like, (3, 3)
"""

ERIMM_RGB_COLOURSPACE = RGB_Colourspace(
    'ERIMM RGB',
    ERIMM_RGB_PRIMARIES,
    ERIMM_RGB_WHITEPOINT,
    ERIMM_RGB_WHITEPOINT_NAME,
    ERIMM_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ERIMM_RGB_MATRIX,
    log_encoding_ERIMMRGB,
    log_decoding_ERIMMRGB,
)
ERIMM_RGB_COLOURSPACE.__doc__ = """
*ERIMM RGB* colourspace.

References
----------
:cite:`Spaulding2000b`

ERIMM_RGB_COLOURSPACE : RGB_Colourspace
"""

PROPHOTO_RGB_PRIMARIES = ROMM_RGB_PRIMARIES
"""
*ProPhoto RGB* colourspace primaries.

PROPHOTO_RGB_PRIMARIES : ndarray, (3, 2)
"""

PROPHOTO_RGB_WHITEPOINT_NAME = ROMM_RGB_WHITEPOINT_NAME
"""
*ProPhoto RGB* colourspace whitepoint name.

PROPHOTO_RGB_WHITEPOINT_NAME : unicode
"""

PROPHOTO_RGB_WHITEPOINT = ROMM_RGB_WHITEPOINT
"""
*ProPhoto RGB* colourspace whitepoint.

PROPHOTO_RGB_WHITEPOINT : ndarray
"""

PROPHOTO_RGB_TO_XYZ_MATRIX = ROMM_RGB_TO_XYZ_MATRIX
"""
*ProPhoto RGB* colourspace to *CIE XYZ* tristimulus values matrix.

PROPHOTO_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_PROPHOTO_RGB_MATRIX = XYZ_TO_ROMM_RGB_MATRIX
"""
*CIE XYZ* tristimulus values to *ProPhoto RGB* colourspace matrix.

XYZ_TO_PROPHOTO_RGB_MATRIX : array_like, (3, 3)
"""

PROPHOTO_RGB_COLOURSPACE = RGB_Colourspace(
    'ProPhoto RGB',
    PROPHOTO_RGB_PRIMARIES,
    PROPHOTO_RGB_WHITEPOINT,
    PROPHOTO_RGB_WHITEPOINT_NAME,
    PROPHOTO_RGB_TO_XYZ_MATRIX,
    XYZ_TO_PROPHOTO_RGB_MATRIX,
    cctf_encoding_ProPhotoRGB,
    cctf_decoding_ProPhotoRGB,
)
PROPHOTO_RGB_COLOURSPACE.__doc__ = """
*ProPhoto RGB* colourspace, an alias colourspace for *ROMM RGB*.

References
----------
:cite:`ANSI2003a`, :cite:`Spaulding2000b`

PROPHOTO_RGB_COLOURSPACE : RGB_Colourspace
"""
