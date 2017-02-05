#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE RGB Colourspace
===================

Defines the *CIE RGB* colourspace:

-   :attr:`CIE_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Fairman, H. S., Brill, M. H., & Hemmendinger, H. (1997). How the CIE
        1931 color-matching functions were derived from Wright-Guild data.
        Color Research & …, 22(1), 11–23. Retrieved from
        http://doi.wiley.com/10.1002/%28SICI%291520-6378%28199702%2922%3A1\
%3C11%3A%3AAID-COL4%3E3.0.CO%3B2-7
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import RGB_Colourspace, gamma_function

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CIE_RGB_PRIMARIES',
           'CIE_RGB_ILLUMINANT',
           'CIE_RGB_WHITEPOINT',
           'CIE_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_CIE_RGB_MATRIX',
           'CIE_RGB_COLOURSPACE']

CIE_RGB_PRIMARIES = np.array(
    [[0.734742840005998, 0.265257159994002],
     [0.273779033824958, 0.717477700256116],
     [0.166555629580280, 0.008910726182545]])
"""
*CIE RGB* colourspace primaries.

CIE_RGB_PRIMARIES : ndarray, (3, 2)

Notes
-----
-   *CIE RGB* colourspace primaries were computed using
    :attr:`CIE_RGB_TO_XYZ_MATRIX` attribute and
    :func:`colour.primaries_whitepoint` definition.
"""

CIE_RGB_ILLUMINANT = 'E'
"""
*CIE RGB* colourspace whitepoint name as illuminant.

CIE_RGB_ILLUMINANT : unicode
"""

CIE_RGB_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][CIE_RGB_ILLUMINANT])
"""
*CIE RGB* colourspace whitepoint.

CIE_RGB_WHITEPOINT : ndarray
"""

CIE_RGB_TO_XYZ_MATRIX = np.array(
    [[0.4900, 0.3100, 0.2000],
     [0.1769, 0.8124, 0.0107],
     [0.0000, 0.0099, 0.9901]])
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
    CIE_RGB_ILLUMINANT,
    CIE_RGB_TO_XYZ_MATRIX,
    XYZ_TO_CIE_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2))
"""
*CIE RGB* colourspace.

CIE_RGB_COLOURSPACE : RGB_Colourspace
"""
