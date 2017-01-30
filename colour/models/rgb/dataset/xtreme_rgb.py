#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Xtreme RGB Colourspace
======================

Defines the *Xtreme RGB* colourspace:

-   :attr:`XTREME_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  HutchColor. (n.d.). XtremeRGB (4 K). Retrieved from
        http://www.hutchcolor.com/profiles/XtremeRGB.zip
"""

from __future__ import division, unicode_literals

import numpy as np
from functools import partial

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    gamma_function,
    normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XTREME_RGB_PRIMARIES',
           'XTREME_RGB_ILLUMINANT',
           'XTREME_RGB_WHITEPOINT',
           'XTREME_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_XTREME_RGB_MATRIX',
           'XTREME_RGB_COLOURSPACE']

XTREME_RGB_PRIMARIES = np.array(
    [[1.0, 0.0],
     [0.0, 1.0],
     [0.0, 0.0]])
"""
*Xtreme RGB* colourspace primaries.

XTREME_RGB_PRIMARIES : ndarray, (3, 2)
"""

XTREME_RGB_ILLUMINANT = 'D50'
"""
*Xtreme RGB* colourspace whitepoint name as illuminant.

XTREME_RGB_WHITEPOINT : unicode
"""

XTREME_RGB_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][XTREME_RGB_ILLUMINANT])
"""
*Xtreme RGB* colourspace whitepoint.

XTREME_RGB_WHITEPOINT : ndarray
"""

XTREME_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    XTREME_RGB_PRIMARIES, XTREME_RGB_WHITEPOINT)
"""
*Xtreme RGB* colourspace to *CIE XYZ* tristimulus values matrix.

XTREME_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_XTREME_RGB_MATRIX = np.linalg.inv(XTREME_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Xtreme RGB* colourspace matrix.

XYZ_TO_XTREME_RGB_MATRIX : array_like, (3, 3)
"""

XTREME_RGB_COLOURSPACE = RGB_Colourspace(
    'Xtreme RGB',
    XTREME_RGB_PRIMARIES,
    XTREME_RGB_WHITEPOINT,
    XTREME_RGB_ILLUMINANT,
    XTREME_RGB_TO_XYZ_MATRIX,
    XYZ_TO_XTREME_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2))
"""
*Xtreme RGB* colourspace.

XTREME_RGB_COLOURSPACE : RGB_Colourspace
"""
