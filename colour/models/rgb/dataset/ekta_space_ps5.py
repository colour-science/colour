# -*- coding: utf-8 -*-
"""
Ekta Space PS 5 Colourspace
===========================

Defines the *Ekta Space PS 5* colourspace:

-   :attr:`colour.models.EKTA_SPACE_PS_5_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Holmesa` : Holmes, J. (n.d.). Ekta Space PS 5. Retrieved from
    https://www.josephholmes.com/userfiles/Ekta_Space_PS5_JHolmes.zip
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial
from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, gamma_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'EKTA_SPACE_PS_5_PRIMARIES', 'EKTA_SPACE_PS_5_V_WHITEPOINT_NAME',
    'EKTA_SPACE_PS_5_WHITEPOINT', 'EKTA_SPACE_PS_5_TO_XYZ_MATRIX',
    'XYZ_TO_EKTA_SPACE_PS_5_MATRIX', 'EKTA_SPACE_PS_5_COLOURSPACE'
]

EKTA_SPACE_PS_5_PRIMARIES = np.array([
    [0.694736842105263, 0.305263157894737],
    [0.260000000000000, 0.700000000000000],
    [0.109728506787330, 0.004524886877828],
])
"""
*Ekta Space PS 5* colourspace primaries.

EKTA_SPACE_PS_5_PRIMARIES : ndarray, (3, 2)
"""

EKTA_SPACE_PS_5_V_WHITEPOINT_NAME = 'D50'
"""
*Ekta Space PS 5* colourspace whitepoint name.

EKTA_SPACE_PS_5_V_WHITEPOINT_NAME : unicode
"""

EKTA_SPACE_PS_5_WHITEPOINT = (ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][EKTA_SPACE_PS_5_V_WHITEPOINT_NAME])
"""
*Ekta Space PS 5* colourspace whitepoint.

EKTA_SPACE_PS_5_WHITEPOINT : ndarray
"""

EKTA_SPACE_PS_5_TO_XYZ_MATRIX = normalised_primary_matrix(
    EKTA_SPACE_PS_5_PRIMARIES, EKTA_SPACE_PS_5_WHITEPOINT)
"""
*Ekta Space PS 5* colourspace to *CIE XYZ* tristimulus values matrix.

EKTA_SPACE_PS_5_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_EKTA_SPACE_PS_5_MATRIX = np.linalg.inv(EKTA_SPACE_PS_5_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Ekta Space PS 5* colourspace matrix.

XYZ_TO_EKTA_SPACE_PS_5_MATRIX : array_like, (3, 3)
"""

EKTA_SPACE_PS_5_COLOURSPACE = RGB_Colourspace(
    'Ekta Space PS 5',
    EKTA_SPACE_PS_5_PRIMARIES,
    EKTA_SPACE_PS_5_WHITEPOINT,
    EKTA_SPACE_PS_5_V_WHITEPOINT_NAME,
    EKTA_SPACE_PS_5_TO_XYZ_MATRIX,
    XYZ_TO_EKTA_SPACE_PS_5_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
EKTA_SPACE_PS_5_COLOURSPACE.__doc__ = """
*Ekta Space PS 5* colourspace.

References
----------
:cite:`Holmesa`

EKTA_SPACE_PS_5_COLOURSPACE : RGB_Colourspace
"""
