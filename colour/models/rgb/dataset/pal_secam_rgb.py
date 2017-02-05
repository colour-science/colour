#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pal/Secam RGB Colourspace
=========================

Defines the *Pal/Secam RGB* colourspace:

-   :attr:`PAL_SECAM_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (1998). CONVENTIONAL TELEVISION
        SYSTEMS. In Recommendation ITU-R BT.470-6 (pp. 1–36). Retrieved from
        http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.470-6-199811-S!!PDF-E.pdf
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

__all__ = ['PAL_SECAM_RGB_PRIMARIES',
           'PAL_SECAM_RGB_ILLUMINANT',
           'PAL_SECAM_RGB_WHITEPOINT',
           'PAL_SECAM_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_PAL_SECAM_RGB_MATRIX',
           'PAL_SECAM_RGB_COLOURSPACE']

PAL_SECAM_RGB_PRIMARIES = np.array(
    [[0.64, 0.33],
     [0.29, 0.60],
     [0.15, 0.06]])
"""
*Pal/Secam RGB* colourspace primaries.

PAL_SECAM_RGB_PRIMARIES : ndarray, (3, 2)
"""

PAL_SECAM_RGB_ILLUMINANT = 'D65'
"""
*Pal/Secam RGB* colourspace whitepoint name as illuminant.

PAL_SECAM_RGB_ILLUMINANT : unicode
"""

PAL_SECAM_RGB_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
        PAL_SECAM_RGB_ILLUMINANT])
"""
*Pal/Secam RGB* colourspace whitepoint.

PAL_SECAM_RGB_WHITEPOINT : ndarray
"""

PAL_SECAM_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(
    PAL_SECAM_RGB_PRIMARIES, PAL_SECAM_RGB_WHITEPOINT)
"""
*Pal/Secam RGB* colourspace to *CIE XYZ* tristimulus values matrix.

PAL_SECAM_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_PAL_SECAM_RGB_MATRIX = np.linalg.inv(PAL_SECAM_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Pal/Secam RGB* colourspace matrix.

XYZ_TO_PAL_SECAM_RGB_MATRIX : array_like, (3, 3)
"""

PAL_SECAM_RGB_COLOURSPACE = RGB_Colourspace(
    'Pal/Secam RGB',
    PAL_SECAM_RGB_PRIMARIES,
    PAL_SECAM_RGB_WHITEPOINT,
    PAL_SECAM_RGB_ILLUMINANT,
    PAL_SECAM_RGB_TO_XYZ_MATRIX,
    XYZ_TO_PAL_SECAM_RGB_MATRIX,
    partial(gamma_function, exponent=1 / 2.8),
    partial(gamma_function, exponent=2.8))
"""
*Pal/Secam RGB* colourspace.

PAL_SECAM_RGB_COLOURSPACE : RGB_Colourspace
"""
