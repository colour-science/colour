#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SMPTE-C RGB Colourspace
=======================

Defines the *SMPTE-C RGB* colourspace:

-   :attr:`SMPTE_C_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Society of Motion Picture and Television Engineers. (2004). SMPTE C
        Color Monitor Colorimetry. In RP 145:2004 (Vol. RP 145:200).
        doi:10.5594/S9781614821649
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

__all__ = ['SMPTE_C_RGB_PRIMARIES',
           'SMPTE_C_RGB_ILLUMINANT',
           'SMPTE_C_RGB_WHITEPOINT',
           'SMPTE_C_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_SMPTE_C_RGB_MATRIX',
           'SMPTE_C_RGB_COLOURSPACE']

SMPTE_C_RGB_PRIMARIES = np.array(
    [[0.630, 0.340],
     [0.310, 0.595],
     [0.155, 0.070]])
"""
*SMPTE-C RGB* colourspace primaries.

SMPTE_C_RGB_PRIMARIES : ndarray, (3, 2)
"""

SMPTE_C_RGB_ILLUMINANT = 'D65'
"""
*SMPTE-C RGB* colourspace whitepoint name as illuminant.

SMPTE_C_RGB_ILLUMINANT : unicode
"""

SMPTE_C_RGB_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][SMPTE_C_RGB_ILLUMINANT])
"""
*SMPTE-C RGB* colourspace whitepoint.

SMPTE_C_RGB_WHITEPOINT : ndarray
"""

SMPTE_C_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    SMPTE_C_RGB_PRIMARIES, SMPTE_C_RGB_WHITEPOINT)
"""
*SMPTE-C RGB* colourspace to *CIE XYZ* tristimulus values matrix.

SMPTE_C_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_SMPTE_C_RGB_MATRIX = np.linalg.inv(SMPTE_C_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *SMPTE-C RGB* colourspace matrix.

XYZ_TO_SMPTE_C_RGB_MATRIX : array_like, (3, 3)
"""

SMPTE_C_RGB_COLOURSPACE = RGB_Colourspace(
    'SMPTE-C RGB',
    SMPTE_C_RGB_PRIMARIES,
    SMPTE_C_RGB_WHITEPOINT,
    SMPTE_C_RGB_ILLUMINANT,
    SMPTE_C_RGB_TO_XYZ_MATRIX,
    XYZ_TO_SMPTE_C_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 2.2),
    partial(gamma_function, exponent=2.2))
"""
*SMPTE-C RGB* colourspace.

SMPTE_C_RGB_COLOURSPACE : RGB_Colourspace
"""
