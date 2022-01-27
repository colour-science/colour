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

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    eotf_DCDM,
    normalised_primary_matrix,
    eotf_inverse_DCDM,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_DCDM_XYZ',
    'WHITEPOINT_NAME_DCDM_XYZ',
    'CCS_WHITEPOINT_DCDM_XYZ',
    'MATRIX_DCDM_XYZ_TO_XYZ',
    'MATRIX_XYZ_TO_DCDM_XYZ',
    'RGB_COLOURSPACE_DCDM_XYZ',
]

PRIMARIES_DCDM_XYZ: NDArray = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
])
"""
*DCDM XYZ* colourspace primaries.
"""

WHITEPOINT_NAME_DCDM_XYZ: str = 'E'
"""
*DCDM XYZ* colourspace whitepoint name.
"""

CCS_WHITEPOINT_DCDM_XYZ: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_DCDM_XYZ])
"""
*DCDM XYZ* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_DCDM_XYZ_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_DCDM_XYZ, CCS_WHITEPOINT_DCDM_XYZ)
"""
*DCDM XYZ* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_DCDM_XYZ: NDArray = np.linalg.inv(MATRIX_DCDM_XYZ_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *DCDM XYZ* colourspace matrix.
"""

RGB_COLOURSPACE_DCDM_XYZ: RGB_Colourspace = RGB_Colourspace(
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
"""
