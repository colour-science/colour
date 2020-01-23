# -*- coding: utf-8 -*-
"""
CIE RGB Colourspace
===================

Defines the *CIE RGB* colourspace:

-   :attr:`colour.models.CIE_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Fairman1997` : Fairman, H. S., Brill, M. H., & Hemmendinger,
    H. (1997). How the CIE 1931 color-matching functions were derived from
    Wright-Guild data. Color Research & Application, 22(1), 11-23.
    doi:10.1002/(SICI)1520-6378(199702)22:1<11::AID-COL4>3.0.CO;2-7
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import RGB_Colourspace, gamma_function

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CIE_RGB_PRIMARIES', 'CIE_RGB_WHITEPOINT_NAME', 'CIE_RGB_WHITEPOINT',
    'CIE_RGB_TO_XYZ_MATRIX', 'XYZ_TO_CIE_RGB_MATRIX', 'CIE_RGB_COLOURSPACE'
]

CIE_RGB_PRIMARIES = np.array([
    [0.734742840005998, 0.265257159994002],
    [0.273779033824958, 0.717477700256116],
    [0.166555629580280, 0.008910726182545],
])
"""
*CIE RGB* colourspace primaries.

CIE_RGB_PRIMARIES : ndarray, (3, 2)

Notes
-----
-   *CIE RGB* colourspace primaries were computed using
    :attr:`colour.models.rgb.datasets.cie_rgb.CIE_RGB_TO_XYZ_MATRIX` attribute
    and :func:`colour.primaries_whitepoint` definition.
"""

CIE_RGB_WHITEPOINT_NAME = 'E'
"""
*CIE RGB* colourspace whitepoint name.

CIE_RGB_WHITEPOINT_NAME : unicode
"""

CIE_RGB_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    CIE_RGB_WHITEPOINT_NAME])
"""
*CIE RGB* colourspace whitepoint.

CIE_RGB_WHITEPOINT : ndarray
"""

CIE_RGB_TO_XYZ_MATRIX = np.array([
    [0.4900, 0.3100, 0.2000],
    [0.1769, 0.8124, 0.0107],
    [0.0000, 0.0099, 0.9901],
])
"""
*CIE RGB* colourspace to *CIE XYZ* tristimulus values matrix.

CIE_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_CIE_RGB_MATRIX = np.linalg.inv(CIE_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *CIE RGB* colourspace matrix.

XYZ_TO_CIE_RGB_MATRIX : array_like, (3, 3)
"""

CIE_RGB_COLOURSPACE = RGB_Colourspace(
    'CIE RGB',
    CIE_RGB_PRIMARIES,
    CIE_RGB_WHITEPOINT,
    CIE_RGB_WHITEPOINT_NAME,
    CIE_RGB_TO_XYZ_MATRIX,
    XYZ_TO_CIE_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2),
)
CIE_RGB_COLOURSPACE.__doc__ = """
*CIE RGB* colourspace.

References
----------
:cite:`Fairman1997`

CIE_RGB_COLOURSPACE : RGB_Colourspace
"""
