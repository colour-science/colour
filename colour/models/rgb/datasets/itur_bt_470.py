# -*- coding: utf-8 -*-
"""
ITU-R BT.470 Colourspaces
=========================

Defines the *ITU-R BT.470* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_BT470_525`.
-   :attr:`colour.models.RGB_COLOURSPACE_BT470_625`.

References
----------
-   :cite:`InternationalTelecommunicationUnion1998a` : International
    Telecommunication Union. (1998). Recommendation ITU-R BT.470-6 -
    CONVENTIONAL TELEVISION SYSTEMS (pp. 1-36).
    http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.470-6-199811-S!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, gamma_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_BT470_525', 'CCS_WHITEPOINT_BT470_525',
    'WHITEPOINT_NAME_BT470_525', 'MATRIX_BT470_525_TO_XYZ',
    'MATRIX_XYZ_TO_BT470_525', 'RGB_COLOURSPACE_BT470_525',
    'PRIMARIES_BT470_625', 'CCS_WHITEPOINT_BT470_625',
    'WHITEPOINT_NAME_BT470_625', 'MATRIX_BT470_625_TO_XYZ',
    'MATRIX_XYZ_TO_BT470_625', 'RGB_COLOURSPACE_BT470_625'
]

PRIMARIES_BT470_525 = np.array([
    [0.6700, 0.3300],
    [0.2100, 0.7100],
    [0.1400, 0.0800],
])
"""
*ITU-R BT.470 - 525* colourspace primaries.

PRIMARIES_BT470_525 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BT470_525 = 'C'
"""
*ITU-R BT.470 - 525* colourspace whitepoint name.

WHITEPOINT_NAME_BT470_525 : unicode
"""

CCS_WHITEPOINT_BT470_525 = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BT470_525])
"""
*ITU-R BT.470 - 525* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_BT470_525 : ndarray
"""

MATRIX_BT470_525_TO_XYZ = normalised_primary_matrix(PRIMARIES_BT470_525,
                                                    CCS_WHITEPOINT_BT470_525)
"""
*ITU-R BT.470 - 525* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BT470_525_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BT470_525 = np.linalg.inv(MATRIX_BT470_525_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *ITU-R BT.470 - 525* colourspace matrix.

MATRIX_XYZ_TO_BT470_525 : array_like, (3, 3)
"""

RGB_COLOURSPACE_BT470_525 = RGB_Colourspace(
    'ITU-R BT.470 - 525',
    PRIMARIES_BT470_525,
    CCS_WHITEPOINT_BT470_525,
    WHITEPOINT_NAME_BT470_525,
    MATRIX_BT470_525_TO_XYZ,
    MATRIX_XYZ_TO_BT470_525,
    partial(gamma_function, exponent=1 / 2.8),
    partial(gamma_function, exponent=2.8),
)
RGB_COLOURSPACE_BT470_525.__doc__ = """
*ITU-R BT.470 - 525* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion1998a`

RGB_COLOURSPACE_BT470_525 : RGB_Colourspace
"""

PRIMARIES_BT470_625 = np.array([
    [0.6400, 0.3300],
    [0.2900, 0.6000],
    [0.1500, 0.0600],
])
"""
*ITU-R BT.470 - 625* colourspace primaries.

PRIMARIES_BT470_625 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BT470_625 = 'D65'
"""
*ITU-R BT.470 - 625* colourspace whitepoint name.

WHITEPOINT_NAME_BT470_625 : unicode
"""

CCS_WHITEPOINT_BT470_625 = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BT470_625])
"""
*ITU-R BT.470 - 625* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_BT470_625 : ndarray
"""

MATRIX_BT470_625_TO_XYZ = normalised_primary_matrix(PRIMARIES_BT470_625,
                                                    CCS_WHITEPOINT_BT470_625)
"""
*ITU-R BT.470 - 625* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BT470_625_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BT470_625 = np.linalg.inv(MATRIX_BT470_625_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *ITU-R BT.470 - 625* colourspace matrix.

MATRIX_XYZ_TO_BT470_625 : array_like, (3, 3)
"""

RGB_COLOURSPACE_BT470_625 = RGB_Colourspace(
    'ITU-R BT.470 - 625',
    PRIMARIES_BT470_625,
    CCS_WHITEPOINT_BT470_625,
    WHITEPOINT_NAME_BT470_625,
    MATRIX_BT470_625_TO_XYZ,
    MATRIX_XYZ_TO_BT470_625,
    partial(gamma_function, exponent=1 / 2.8),
    partial(gamma_function, exponent=2.8),
)
RGB_COLOURSPACE_BT470_625.__doc__ = """
*ITU-R BT.470 - 625* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion1998a`

RGB_COLOURSPACE_BT470_625 : RGB_Colourspace
"""
