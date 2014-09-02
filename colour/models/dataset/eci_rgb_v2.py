#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ECI RGB v2 Colourspace
======================

Defines the *ECI RGB v2* colourspace:

-   :attr:`ECI_RGB_V2_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  http://www.eci.org/_media/downloads/icc_profiles_from_eci/ecirgbv20.zip
        (Last accessed 13 April 2014)
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS, lightness_1976, luminance_1976
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ECI_RGB_V2_PRIMARIES',
           'ECI_RGB_V2_WHITEPOINT',
           'ECI_RGB_V2_TO_XYZ_MATRIX',
           'XYZ_TO_ECI_RGB_V2_MATRIX',
           'ECI_RGB_V2_TRANSFER_FUNCTION',
           'ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION',
           'ECI_RGB_V2_COLOURSPACE']

ECI_RGB_V2_PRIMARIES = np.array(
    [[0.67010309278350522, 0.32989690721649484],
     [0.20990566037735847, 0.70990566037735836],
     [0.14006179196704427, 0.080329557157569509]])
"""
*ECI RGB v2* colourspace primaries.

ECI_RGB_V2_PRIMARIES : ndarray, (3, 2)
"""

ECI_RGB_V2_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get('D50')
"""
*ECI RGB v2* colourspace whitepoint.

ECI_RGB_V2_WHITEPOINT : tuple
"""

ECI_RGB_V2_TO_XYZ_MATRIX = normalised_primary_matrix(ECI_RGB_V2_PRIMARIES,
                                                     ECI_RGB_V2_WHITEPOINT)
"""
*ECI RGB v2* colourspace to *CIE XYZ* colourspace matrix.

ECI_RGB_V2_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_ECI_RGB_V2_MATRIX = np.linalg.inv(ECI_RGB_V2_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *ECI RGB v2* colourspace matrix.

XYZ_TO_ECI_RGB_V2_MATRIX : array_like, (3, 3)
"""

ECI_RGB_V2_TRANSFER_FUNCTION = lambda x: lightness_1976(x * 100) / 100
"""
Transfer function from linear to *ECI RGB v2* colourspace.

ECI_RGB_V2_TRANSFER_FUNCTION : object
"""

ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION = lambda x: (
    luminance_1976(x * 100) / 100)
"""
Inverse transfer function from *ECI RGB v2* colourspace to linear.

ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION : object
"""

ECI_RGB_V2_COLOURSPACE = RGB_Colourspace(
    'ECI RGB v2',
    ECI_RGB_V2_PRIMARIES,
    ECI_RGB_V2_WHITEPOINT,
    ECI_RGB_V2_TO_XYZ_MATRIX,
    XYZ_TO_ECI_RGB_V2_MATRIX,
    ECI_RGB_V2_TRANSFER_FUNCTION,
    ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION)
"""
*ECI RGB v2* colourspace.

ECI_RGB_V2_COLOURSPACE : RGB_Colourspace
"""
