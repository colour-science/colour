# -*- coding: utf-8 -*-
"""
Ekta Space PS 5 Colourspace
===========================

Defines the *Ekta Space PS 5* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_EKTA_SPACE_PS_5`.

References
----------
-   :cite:`Holmesa` : Holmes, J. (n.d.). Ekta Space PS 5.
    https://www.josephholmes.com/userfiles/Ekta_Space_PS5_JHolmes.zip
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
    'PRIMARIES_EKTA_SPACE_PS_5', 'WHITEPOINT_NAME_EKTA_SPACE_PS_5_V',
    'CCS_WHITEPOINT_EKTA_SPACE_PS_5', 'MATRIX_EKTA_SPACE_PS_5_TO_XYZ',
    'MATRIX_XYZ_TO_EKTA_SPACE_PS_5', 'RGB_COLOURSPACE_EKTA_SPACE_PS_5'
]

PRIMARIES_EKTA_SPACE_PS_5 = np.array([
    [0.694736842105263, 0.305263157894737],
    [0.260000000000000, 0.700000000000000],
    [0.109728506787330, 0.004524886877828],
])
"""
*Ekta Space PS 5* colourspace primaries.

PRIMARIES_EKTA_SPACE_PS_5 : ndarray, (3, 2)
"""

WHITEPOINT_NAME_EKTA_SPACE_PS_5_V = 'D50'
"""
*Ekta Space PS 5* colourspace whitepoint name.

WHITEPOINT_NAME_EKTA_SPACE_PS_5_V : unicode
"""

CCS_WHITEPOINT_EKTA_SPACE_PS_5 = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_EKTA_SPACE_PS_5_V])
"""
*Ekta Space PS 5* colourspace whitepoint chromaticity coordinates.

CCS_WHITEPOINT_EKTA_SPACE_PS_5 : ndarray
"""

MATRIX_EKTA_SPACE_PS_5_TO_XYZ = normalised_primary_matrix(
    PRIMARIES_EKTA_SPACE_PS_5, CCS_WHITEPOINT_EKTA_SPACE_PS_5)
"""
*Ekta Space PS 5* colourspace to *CIE XYZ* tristimulus values matrix.

MATRIX_EKTA_SPACE_PS_5_TO_XYZ : array_like, (3, 3)
"""

MATRIX_XYZ_TO_EKTA_SPACE_PS_5 = np.linalg.inv(MATRIX_EKTA_SPACE_PS_5_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Ekta Space PS 5* colourspace matrix.

MATRIX_XYZ_TO_EKTA_SPACE_PS_5 : array_like, (3, 3)
"""

RGB_COLOURSPACE_EKTA_SPACE_PS_5 = RGB_Colourspace(
    'Ekta Space PS 5',
    PRIMARIES_EKTA_SPACE_PS_5,
    CCS_WHITEPOINT_EKTA_SPACE_PS_5,
    WHITEPOINT_NAME_EKTA_SPACE_PS_5_V,
    MATRIX_EKTA_SPACE_PS_5_TO_XYZ,
    MATRIX_XYZ_TO_EKTA_SPACE_PS_5,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
RGB_COLOURSPACE_EKTA_SPACE_PS_5.__doc__ = """
*Ekta Space PS 5* colourspace.

References
----------
:cite:`Holmesa`

RGB_COLOURSPACE_EKTA_SPACE_PS_5 : RGB_Colourspace
"""
