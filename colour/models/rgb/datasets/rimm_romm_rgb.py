# -*- coding: utf-8 -*-
"""
RIMM, ROMM and ERIMM Encodings
==============================

Defines the *RIMM, ROMM and ERIMM* encodings:

-   :attr:`colour.models.RGB_COLOURSPACE_ROMM_RGB`.
-   :attr:`colour.models.RGB_COLOURSPACE_RIMM_RGB`.
-   :attr:`colour.models.RGB_COLOURSPACE_ERIMM_RGB`.
-   :attr:`colour.models.RGB_COLOURSPACE_PROPHOTO_RGB`.

References
----------
-   :cite:`ANSI2003a` : ANSI. (2003). Specification of ROMM RGB (pp. 1-2).
    http://www.color.org/ROMMRGB.pdf
-   :cite:`Spaulding2000b` : Spaulding, K. E., Woolfe, G. J., & Giorgianni, E.
    J. (2000). Reference Input/Output Medium Metric RGB Color Encodings
    (RIMM/ROMM RGB) (pp. 1-8). http://www.photo-lovers.org/pdf/color/romm.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    cctf_encoding_ROMMRGB,
    cctf_decoding_ROMMRGB,
    cctf_encoding_RIMMRGB,
    cctf_decoding_RIMMRGB,
    log_encoding_ERIMMRGB,
    log_decoding_ERIMMRGB,
    cctf_encoding_ProPhotoRGB,
    cctf_decoding_ProPhotoRGB,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_ROMM_RGB',
    'WHITEPOINT_NAME_ROMM_RGB',
    'CCS_WHITEPOINT_ROMM_RGB',
    'MATRIX_ROMM_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_ROMM_RGB',
    'RGB_COLOURSPACE_ROMM_RGB',
    'PRIMARIES_RIMM_RGB',
    'WHITEPOINT_NAME_RIMM_RGB',
    'CCS_WHITEPOINT_RIMM_RGB',
    'MATRIX_RIMM_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_RIMM_RGB',
    'RGB_COLOURSPACE_RIMM_RGB',
    'PRIMARIES_ERIMM_RGB',
    'WHITEPOINT_NAME_ERIMM_RGB',
    'CCS_WHITEPOINT_ERIMM_RGB',
    'MATRIX_ERIMM_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_ERIMM_RGB',
    'RGB_COLOURSPACE_ERIMM_RGB',
    'PRIMARIES_PROPHOTO_RGB',
    'WHITEPOINT_NAME_PROPHOTO_RGB',
    'CCS_WHITEPOINT_PROPHOTO_RGB',
    'MATRIX_PROPHOTO_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_PROPHOTO_RGB',
    'RGB_COLOURSPACE_PROPHOTO_RGB',
]

PRIMARIES_ROMM_RGB: NDArray = np.array([
    [0.7347, 0.2653],
    [0.1596, 0.8404],
    [0.0366, 0.0001],
])
"""
*ROMM RGB* colourspace primaries.
"""

WHITEPOINT_NAME_ROMM_RGB: str = 'D50'
"""
*ROMM RGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_ROMM_RGB: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_ROMM_RGB])
"""
*ROMM RGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_ROMM_RGB_TO_XYZ: NDArray = np.array([
    [0.7977, 0.1352, 0.0313],
    [0.2880, 0.7119, 0.0001],
    [0.0000, 0.0000, 0.8249],
])
"""
*ROMM RGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_ROMM_RGB: NDArray = np.array([
    [1.3460, -0.2556, -0.0511],
    [-0.5446, 1.5082, 0.0205],
    [0.0000, 0.0000, 1.2123],
])
"""
*CIE XYZ* tristimulus values to *ROMM RGB* colourspace matrix.
"""

RGB_COLOURSPACE_ROMM_RGB: RGB_Colourspace = RGB_Colourspace(
    'ROMM RGB',
    PRIMARIES_ROMM_RGB,
    CCS_WHITEPOINT_ROMM_RGB,
    WHITEPOINT_NAME_ROMM_RGB,
    MATRIX_ROMM_RGB_TO_XYZ,
    MATRIX_XYZ_TO_ROMM_RGB,
    cctf_encoding_ROMMRGB,
    cctf_decoding_ROMMRGB,
)
RGB_COLOURSPACE_ROMM_RGB.__doc__ = """
*ROMM RGB* colourspace.

References
----------
:cite:`ANSI2003a`, :cite:`Spaulding2000b`
"""

PRIMARIES_RIMM_RGB: NDArray = PRIMARIES_ROMM_RGB
"""
*RIMM RGB* colourspace primaries.
"""

WHITEPOINT_NAME_RIMM_RGB: str = WHITEPOINT_NAME_ROMM_RGB
"""
*RIMM RGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_RIMM_RGB: NDArray = CCS_WHITEPOINT_ROMM_RGB
"""
*RIMM RGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_RIMM_RGB_TO_XYZ: NDArray = MATRIX_ROMM_RGB_TO_XYZ
"""
*RIMM RGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_RIMM_RGB: NDArray = MATRIX_XYZ_TO_ROMM_RGB
"""
*CIE XYZ* tristimulus values to *RIMM RGB* colourspace matrix.
"""

RGB_COLOURSPACE_RIMM_RGB: RGB_Colourspace = RGB_Colourspace(
    'RIMM RGB',
    PRIMARIES_RIMM_RGB,
    CCS_WHITEPOINT_RIMM_RGB,
    WHITEPOINT_NAME_RIMM_RGB,
    MATRIX_RIMM_RGB_TO_XYZ,
    MATRIX_XYZ_TO_RIMM_RGB,
    cctf_encoding_RIMMRGB,
    cctf_decoding_RIMMRGB,
)
RGB_COLOURSPACE_RIMM_RGB.__doc__ = """
*RIMM RGB* colourspace. In cases in which it is necessary to identify a
specific precision level, the notation *RIMM8 RGB*, *RIMM12 RGB* and
*RIMM16 RGB* is used.

References
----------
:cite:`Spaulding2000b`
"""

PRIMARIES_ERIMM_RGB: NDArray = PRIMARIES_ROMM_RGB
"""
*ERIMM RGB* colourspace primaries.
"""

WHITEPOINT_NAME_ERIMM_RGB: str = WHITEPOINT_NAME_ROMM_RGB
"""
*ERIMM RGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_ERIMM_RGB: NDArray = CCS_WHITEPOINT_ROMM_RGB
"""
*ERIMM RGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_ERIMM_RGB_TO_XYZ: NDArray = MATRIX_ROMM_RGB_TO_XYZ
"""
*ERIMM RGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_ERIMM_RGB: NDArray = MATRIX_XYZ_TO_ROMM_RGB
"""
*CIE XYZ* tristimulus values to *ERIMM RGB* colourspace matrix.
"""

RGB_COLOURSPACE_ERIMM_RGB: RGB_Colourspace = RGB_Colourspace(
    'ERIMM RGB',
    PRIMARIES_ERIMM_RGB,
    CCS_WHITEPOINT_ERIMM_RGB,
    WHITEPOINT_NAME_ERIMM_RGB,
    MATRIX_ERIMM_RGB_TO_XYZ,
    MATRIX_XYZ_TO_ERIMM_RGB,
    log_encoding_ERIMMRGB,
    log_decoding_ERIMMRGB,
)
RGB_COLOURSPACE_ERIMM_RGB.__doc__ = """
*ERIMM RGB* colourspace.

References
----------
:cite:`Spaulding2000b`
"""

PRIMARIES_PROPHOTO_RGB: NDArray = PRIMARIES_ROMM_RGB
"""
*ProPhoto RGB* colourspace primaries.
"""

WHITEPOINT_NAME_PROPHOTO_RGB: str = WHITEPOINT_NAME_ROMM_RGB
"""
*ProPhoto RGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_PROPHOTO_RGB: NDArray = CCS_WHITEPOINT_ROMM_RGB
"""
*ProPhoto RGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_PROPHOTO_RGB_TO_XYZ: NDArray = MATRIX_ROMM_RGB_TO_XYZ
"""
*ProPhoto RGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_PROPHOTO_RGB: NDArray = MATRIX_XYZ_TO_ROMM_RGB
"""
*CIE XYZ* tristimulus values to *ProPhoto RGB* colourspace matrix.
"""

RGB_COLOURSPACE_PROPHOTO_RGB: RGB_Colourspace = RGB_Colourspace(
    'ProPhoto RGB',
    PRIMARIES_PROPHOTO_RGB,
    CCS_WHITEPOINT_PROPHOTO_RGB,
    WHITEPOINT_NAME_PROPHOTO_RGB,
    MATRIX_PROPHOTO_RGB_TO_XYZ,
    MATRIX_XYZ_TO_PROPHOTO_RGB,
    cctf_encoding_ProPhotoRGB,
    cctf_decoding_ProPhotoRGB,
)
RGB_COLOURSPACE_PROPHOTO_RGB.__doc__ = """
*ProPhoto RGB* colourspace, an alias colourspace for *ROMM RGB*.

References
----------
:cite:`ANSI2003a`, :cite:`Spaulding2000b`
"""
