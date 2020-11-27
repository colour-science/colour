# -*- coding: utf-8 -*-
"""
SMPTE 240M Colourspace
======================

Defines the *SMPTE 240M* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_SMPTE_240M`.

References
----------
-   :cite:`SocietyofMotionPictureandTelevisionEngineers1999b` : Society of
    Motion Picture and Television Engineers. (1999). ANSI/SMPTE 240M-1995 -
    Signal Parameters - 1125-Line High-Definition Production Systems (pp. 1-7).
    http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/\
SMPTE%20normes%20et%20confs/s240m.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, normalised_primary_matrix,
                               oetf_SMPTE240M, eotf_SMPTE240M)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_SMPTE_240M', 'WHITEPOINT_NAME_SMPTE_240M',
    'CCS_WHITEPOINT_SMPTE_240M', 'MATRIX_SMPTE_240M_TO_XYZ',
    'MATRIX_XYZ_TO_SMPTE_240M', 'RGB_COLOURSPACE_SMPTE_240M'
]

PRIMARIES_SMPTE_240M = np.array([
    [0.6300, 0.3400],
    [0.3100, 0.5950],
    [0.1550, 0.0700],
])
"""
*SMPTE 240M* colourspace primaries.

PRIMARIES_SMPTE_240M : ndarray, (3, 2)
"""

WHITEPOINT_NAME_SMPTE_240M = 'D65'
"""
*SMPTE 240M* colourspace whitepoint name.

WHITEPOINT_NAME_SMPTE_240M : unicode
"""

CCS_WHITEPOINT_SMPTE_240M = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_SMPTE_240M])
"""
*SMPTE 240M* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_SMPTE_240M : ndarray
"""

MATRIX_SMPTE_240M_TO_XYZ = normalised_primary_matrix(
    PRIMARIES_SMPTE_240M, CCS_WHITEPOINT_SMPTE_240M)
"""
*SMPTE 240M* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_SMPTE_240M_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_SMPTE_240M = np.linalg.inv(MATRIX_SMPTE_240M_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *SMPTE 240M* colourspace matrix.

MATRIX_XYZ_TO_SMPTE_240M : array_like, (3, 3)
"""

RGB_COLOURSPACE_SMPTE_240M = RGB_Colourspace(
    'SMPTE 240M',
    PRIMARIES_SMPTE_240M,
    CCS_WHITEPOINT_SMPTE_240M,
    WHITEPOINT_NAME_SMPTE_240M,
    MATRIX_SMPTE_240M_TO_XYZ,
    MATRIX_XYZ_TO_SMPTE_240M,
    oetf_SMPTE240M,
    eotf_SMPTE240M,
)
RGB_COLOURSPACE_SMPTE_240M.__doc__ = """
*SMPTE 240M* colourspace.

References
----------
:cite:`SocietyofMotionPictureandTelevisionEngineers1999b`,

RGB_COLOURSPACE_SMPTE_240M : RGB_Colourspace
"""
