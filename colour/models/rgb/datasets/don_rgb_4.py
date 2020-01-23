# -*- coding: utf-8 -*-
"""
Don RGB 4 Colourspace
=====================

Defines the *Don RGB 4* colourspace:

-   :attr:`colour.models.DON_RGB_4_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`HutchColorg` : HutchColor. (n.d.). DonRGB4 (4 K). Retrieved from
    http://www.hutchcolor.com/profiles/DonRGB4.zip
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, gamma_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'DON_RGB_4_PRIMARIES', 'DON_RGB_4_WHITEPOINT_NAME', 'DON_RGB_4_WHITEPOINT',
    'DON_RGB_4_TO_XYZ_MATRIX', 'XYZ_TO_DON_RGB_4_MATRIX',
    'DON_RGB_4_COLOURSPACE'
]

DON_RGB_4_PRIMARIES = np.array([
    [0.696120689655172, 0.299568965517241],
    [0.214682981090100, 0.765294771968854],
    [0.129937629937630, 0.035343035343035],
])
"""
*Don RGB 4* colourspace primaries.

DON_RGB_4_PRIMARIES : ndarray, (3, 2)
"""

DON_RGB_4_WHITEPOINT_NAME = 'D50'
"""
*Don RGB 4* colourspace whitepoint name.

DON_RGB_4_WHITEPOINT_NAME : unicode
"""

DON_RGB_4_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    DON_RGB_4_WHITEPOINT_NAME])
"""
*Don RGB 4* colourspace whitepoint.

DON_RGB_4_WHITEPOINT : ndarray
"""

DON_RGB_4_TO_XYZ_MATRIX = normalised_primary_matrix(DON_RGB_4_PRIMARIES,
                                                    DON_RGB_4_WHITEPOINT)
"""
*Don RGB 4* colourspace to *CIE XYZ* tristimulus values matrix.

DON_RGB_4_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DON_RGB_4_MATRIX = np.linalg.inv(DON_RGB_4_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Don RGB 4* colourspace matrix.

XYZ_TO_DON_RGB_4_MATRIX : array_like, (3, 3)
"""

DON_RGB_4_COLOURSPACE = RGB_Colourspace(
    'Don RGB 4',
    DON_RGB_4_PRIMARIES,
    DON_RGB_4_WHITEPOINT,
    DON_RGB_4_WHITEPOINT_NAME,
    DON_RGB_4_TO_XYZ_MATRIX,
    XYZ_TO_DON_RGB_4_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
DON_RGB_4_COLOURSPACE.__doc__ = """
*Don RGB 4* colourspace.

References
----------
:cite:`HutchColorg`

DON_RGB_4_COLOURSPACE : RGB_Colourspace
"""
