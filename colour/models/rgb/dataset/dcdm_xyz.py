# -*- coding: utf-8 -*-
"""
Digital Cinema Distribution Master (DCDM) XYZ Colourspace
=========================================================

Defines the *DCDM XYZ* colourspace:

-   :attr:`colour.models.DCDM_XYZ_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`DigitalCinemaInitiatives2007b` : Digital Cinema Initiatives. (2007).
    Digital Cinema System Specification - Version 1.1. Retrieved from
    http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, eotf_DCDM,
                               normalised_primary_matrix, eotf_reverse_DCDM)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'DCDM_XYZ_PRIMARIES', 'DCDM_XYZ_WHITEPOINT_NAME', 'DCDM_XYZ_WHITEPOINT',
    'DCDM_XYZ_TO_XYZ_MATRIX', 'XYZ_TO_DCDM_XYZ_MATRIX', 'DCDM_XYZ_COLOURSPACE'
]

DCDM_XYZ_PRIMARIES = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
])
"""
*DCDM XYZ* colourspace primaries.

DCDM_XYZ_PRIMARIES : ndarray, (3, 2)
"""

DCDM_XYZ_WHITEPOINT_NAME = 'E'
"""
*DCDM XYZ* colourspace whitepoint name.

DCDM_XYZ_WHITEPOINT_NAME : unicode
"""

DCDM_XYZ_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    DCDM_XYZ_WHITEPOINT_NAME])
"""
*DCDM XYZ* colourspace whitepoint.

DCDM_XYZ_WHITEPOINT : ndarray
"""

DCDM_XYZ_TO_XYZ_MATRIX = normalised_primary_matrix(DCDM_XYZ_PRIMARIES,
                                                   DCDM_XYZ_WHITEPOINT)
"""
*DCDM XYZ* colourspace to *CIE XYZ* tristimulus values matrix.

DCDM_XYZ_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DCDM_XYZ_MATRIX = np.linalg.inv(DCDM_XYZ_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *DCDM XYZ* colourspace matrix.

XYZ_TO_DCDM_XYZ_MATRIX : array_like, (3, 3)
"""

DCDM_XYZ_COLOURSPACE = RGB_Colourspace(
    'DCDM XYZ',
    DCDM_XYZ_PRIMARIES,
    DCDM_XYZ_WHITEPOINT,
    DCDM_XYZ_WHITEPOINT_NAME,
    DCDM_XYZ_TO_XYZ_MATRIX,
    XYZ_TO_DCDM_XYZ_MATRIX,
    eotf_reverse_DCDM,
    eotf_DCDM,
)
DCDM_XYZ_COLOURSPACE.__doc__ = """
*DCDM XYZ* colourspace.

References
----------
:cite:`DigitalCinemaInitiatives2007b`

DCDM_XYZ_COLOURSPACE : RGB_Colourspace
"""
