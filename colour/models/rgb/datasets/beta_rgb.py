# -*- coding: utf-8 -*-
"""
Beta RGB Colourspace
====================

Defines the *Beta RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_BETA_RGB`.

References
----------
-   :cite:`Lindbloom2014a` : Lindbloom, B. (2014). RGB Working Space
    Information. Retrieved April 11, 2014, from
    http://www.brucelindbloom.com/WorkingSpaceInfo.html
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
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
    'PRIMARIES_BETA_RGB', 'WHITEPOINT_NAME_BETA_RGB',
    'CCS_WHITEPOINT_BETA_RGB', 'MATRIX_BETA_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_BETA_RGB', 'RGB_COLOURSPACE_BETA_RGB'
]

PRIMARIES_BETA_RGB = np.array([
    [0.6888, 0.3112],
    [0.1986, 0.7551],
    [0.1265, 0.0352],
])
"""
*Beta RGB* colourspace primaries.

PRIMARIES_BETA_RGB : ndarray, (3, 2)
"""

WHITEPOINT_NAME_BETA_RGB = 'D50'
"""
*Beta RGB* colourspace whitepoint name.

WHITEPOINT_NAME_BETA_RGB : unicode
"""

CCS_WHITEPOINT_BETA_RGB = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_BETA_RGB])
"""
*Beta RGB* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_BETA_RGB : ndarray
"""

MATRIX_BETA_RGB_TO_XYZ = normalised_primary_matrix(PRIMARIES_BETA_RGB,
                                                   CCS_WHITEPOINT_BETA_RGB)
"""
*Beta RGB* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_BETA_RGB_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_BETA_RGB = np.linalg.inv(MATRIX_BETA_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Beta RGB* colourspace matrix.

MATRIX_XYZ_TO_BETA_RGB : array_like, (3, 3)
"""

RGB_COLOURSPACE_BETA_RGB = RGB_Colourspace(
    'Beta RGB',
    PRIMARIES_BETA_RGB,
    CCS_WHITEPOINT_BETA_RGB,
    WHITEPOINT_NAME_BETA_RGB,
    MATRIX_BETA_RGB_TO_XYZ,
    MATRIX_XYZ_TO_BETA_RGB,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_BETA_RGB.__doc__ = """
*Beta RGB* colourspace.

References
----------
:cite:`Lindbloom2014a`

RGB_COLOURSPACE_BETA_RGB : RGB_Colourspace
"""
