# -*- coding: utf-8 -*-
"""
ITU-R BT.709 Colourspace
========================

Defines the *ITU-R BT.709* colourspace:

-   :attr:`colour.models.BT709_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`InternationalTelecommunicationUnion2015i` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.709-6 - Parameter
    values for the HDTV standards for production and international programme
    exchange BT Series Broadcasting service. Retrieved from
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, oetf_BT709, oetf_inverse_BT709,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'BT709_PRIMARIES', 'BT709_WHITEPOINT', 'BT709_WHITEPOINT_NAME',
    'BT709_TO_XYZ_MATRIX', 'XYZ_TO_BT709_MATRIX', 'BT709_COLOURSPACE'
]

BT709_PRIMARIES = np.array([
    [0.6400, 0.3300],
    [0.3000, 0.6000],
    [0.1500, 0.0600],
])
"""
*ITU-R BT.709* colourspace primaries.

BT709_PRIMARIES : ndarray, (3, 2)
"""

BT709_WHITEPOINT_NAME = 'D65'
"""
*ITU-R BT.709* colourspace whitepoint name.

BT709_WHITEPOINT_NAME : unicode
"""

BT709_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][BT709_WHITEPOINT_NAME])
"""
*ITU-R BT.709* colourspace whitepoint.

BT709_WHITEPOINT : ndarray
"""

BT709_TO_XYZ_MATRIX = normalised_primary_matrix(BT709_PRIMARIES,
                                                BT709_WHITEPOINT)
"""
*ITU-R BT.709* colourspace to *CIE XYZ* tristimulus values matrix.

BT709_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_BT709_MATRIX = np.linalg.inv(BT709_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *ITU-R BT.709* colourspace matrix.

XYZ_TO_BT709_MATRIX : array_like, (3, 3)
"""

BT709_COLOURSPACE = RGB_Colourspace(
    'ITU-R BT.709',
    BT709_PRIMARIES,
    BT709_WHITEPOINT,
    BT709_WHITEPOINT_NAME,
    BT709_TO_XYZ_MATRIX,
    XYZ_TO_BT709_MATRIX,
    oetf_BT709,
    oetf_inverse_BT709,
)
BT709_COLOURSPACE.__doc__ = """
*ITU-R BT.709* colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion2015i`

BT709_COLOURSPACE : RGB_Colourspace
"""
