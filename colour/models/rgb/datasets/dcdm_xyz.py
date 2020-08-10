# -*- coding: utf-8 -*-
"""
Digital Cinema Distribution Master (DCDM) XYZ Colourspace
=========================================================

Defines the *DCDM XYZ* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_DCDM_XYZ`.

References
----------
-   :cite:`DigitalCinemaInitiatives2007b` : Digital Cinema Initiatives. (2007).
    Digital Cinema System Specification - Version 1.1.
    http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, eotf_DCDM,
                               normalised_primary_matrix, eotf_inverse_DCDM)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_DCDM_XYZ', 'WHITEPOINT_NAME_DCDM_XYZ',
    'CCS_WHITEPOINT_DCDM_XYZ', 'MATRIX_DCDM_XYZ_TO_XYZ',
    'MATRIX_XYZ_TO_DCDM_XYZ', 'RGB_COLOURSPACE_DCDM_XYZ'
]

PRIMARIES_DCDM_XYZ = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
])
"""
*DCDM XYZ* colourspace primaries.

PRIMARIES_DCDM_XYZ : ndarray, (3, 2)
"""

WHITEPOINT_NAME_DCDM_XYZ = 'E'
"""
*DCDM XYZ* colourspace whitepoint name.

WHITEPOINT_NAME_DCDM_XYZ : unicode
"""

CCS_WHITEPOINT_DCDM_XYZ = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_DCDM_XYZ])
"""
*DCDM XYZ* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_DCDM_XYZ : ndarray
"""

MATRIX_DCDM_XYZ_TO_XYZ = normalised_primary_matrix(PRIMARIES_DCDM_XYZ,
                                                   CCS_WHITEPOINT_DCDM_XYZ)
"""
*DCDM XYZ* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_DCDM_XYZ_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_DCDM_XYZ = np.linalg.inv(MATRIX_DCDM_XYZ_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *DCDM XYZ* colourspace matrix.

MATRIX_XYZ_TO_DCDM_XYZ : array_like, (3, 3)
"""

RGB_COLOURSPACE_DCDM_XYZ = RGB_Colourspace(
    'DCDM XYZ',
    PRIMARIES_DCDM_XYZ,
    CCS_WHITEPOINT_DCDM_XYZ,
    WHITEPOINT_NAME_DCDM_XYZ,
    MATRIX_DCDM_XYZ_TO_XYZ,
    MATRIX_XYZ_TO_DCDM_XYZ,
    eotf_inverse_DCDM,
    eotf_DCDM,
)
RGB_COLOURSPACE_DCDM_XYZ.__doc__ = """
*DCDM XYZ* colourspace.

References
----------
:cite:`DigitalCinemaInitiatives2007b`

RGB_COLOURSPACE_DCDM_XYZ : RGB_Colourspace
"""
