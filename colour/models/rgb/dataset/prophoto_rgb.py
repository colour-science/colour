#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProPhoto RGB Colourspace
========================

Defines the *ProPhoto RGB* colourspace:

-   :attr:`PROPHOTO_RGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Spaulding, K. E., Woolfe, G. J., & Giorgianni, E. J. (2000). Reference
        Input/Output Medium Metric RGB Color Encodings (RIMM/ROMM RGB), 1–8.
        Retrieved from http://www.photo-lovers.org/pdf/color/romm.pdf
.. [2]  ANSI. (2003). Specification of ROMM RGB. Retrieved from
        http://www.color.org/ROMMRGB.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    oetf_ProPhotoRGB,
    eotf_ProPhotoRGB)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['PROPHOTO_RGB_PRIMARIES',
           'PROPHOTO_RGB_ILLUMINANT',
           'PROPHOTO_RGB_WHITEPOINT',
           'PROPHOTO_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_PROPHOTO_RGB_MATRIX',
           'PROPHOTO_RGB_COLOURSPACE']

PROPHOTO_RGB_PRIMARIES = np.array(
    [[0.7347, 0.2653],
     [0.1596, 0.8404],
     [0.0366, 0.0001]])
"""
*ProPhoto RGB* colourspace primaries.

PROPHOTO_RGB_PRIMARIES : ndarray, (3, 2)
"""

PROPHOTO_RGB_ILLUMINANT = 'D50'
"""
*ProPhoto RGB* colourspace whitepoint name as illuminant.

PROPHOTO_RGB_ILLUMINANT : unicode
"""

PROPHOTO_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(PROPHOTO_RGB_ILLUMINANT)
"""
*ProPhoto RGB* colourspace whitepoint.

PROPHOTO_RGB_WHITEPOINT : tuple
"""

PROPHOTO_RGB_TO_XYZ_MATRIX = np.array(
    [[7.97667235e-01, 1.35192231e-01, 3.13525290e-02],
     [2.88037454e-01, 7.11876883e-01, 8.56626476e-05],
     [0.00000000e+00, 0.00000000e+00, 8.25188285e-01]])
"""
*ProPhoto RGB* colourspace to *CIE XYZ* tristimulus values matrix.

PROPHOTO_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_PROPHOTO_RGB_MATRIX = np.linalg.inv(PROPHOTO_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *ProPhoto RGB* colourspace matrix.

XYZ_TO_PROPHOTO_RGB_MATRIX : array_like, (3, 3)
"""


PROPHOTO_RGB_COLOURSPACE = RGB_Colourspace(
    'ProPhoto RGB',
    PROPHOTO_RGB_PRIMARIES,
    PROPHOTO_RGB_WHITEPOINT,
    PROPHOTO_RGB_ILLUMINANT,
    PROPHOTO_RGB_TO_XYZ_MATRIX,
    XYZ_TO_PROPHOTO_RGB_MATRIX,
    oetf_ProPhotoRGB,
    eotf_ProPhotoRGB)
"""
*ProPhoto RGB* colourspace.

PROPHOTO_RGB_COLOURSPACE : RGB_Colourspace
"""
