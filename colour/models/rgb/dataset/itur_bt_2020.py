# -*- coding: utf-8 -*-
"""
ITU-R BT.2020 Colourspace
=========================

Defines the *ITU-R BT.2020* colourspace:

-   :attr:`colour.models.BT2020_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`InternationalTelecommunicationUnion2015h` : International
    Telecommunication Union. (2015). Recommendation ITU-R BT.2020 - Parameter
    values for ultra-high definition television systems for production and
    international programme exchange. Retrieved from https://www.itu.int/\
dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-2-201510-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, normalised_primary_matrix,
                               oetf_BT2020, eotf_BT2020)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'BT2020_PRIMARIES', 'BT2020_ILLUMINANT', 'BT2020_WHITEPOINT',
    'BT2020_TO_XYZ_MATRIX', 'XYZ_TO_BT2020_MATRIX', 'BT2020_COLOURSPACE'
]

BT2020_PRIMARIES = np.array([
    [0.708, 0.292],
    [0.170, 0.797],
    [0.131, 0.046],
])
"""
*ITU-R BT.2020* colourspace primaries.

BT2020_PRIMARIES : ndarray, (3, 2)
"""

BT2020_ILLUMINANT = 'D65'
"""
*ITU-R BT.2020* colourspace whitepoint name as illuminant.

BT2020_ILLUMINANT : unicode
"""

BT2020_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][BT2020_ILLUMINANT])
"""
*ITU-R BT.2020* colourspace whitepoint.

BT2020_WHITEPOINT : ndarray
"""

BT2020_TO_XYZ_MATRIX = normalised_primary_matrix(BT2020_PRIMARIES,
                                                 BT2020_WHITEPOINT)
"""
*ITU-R BT.2020* colourspace to *CIE XYZ* tristimulus values matrix.

BT2020_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_BT2020_MATRIX = np.linalg.inv(BT2020_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *ITU-R BT.2020* colourspace matrix.

XYZ_TO_BT2020_MATRIX : array_like, (3, 3)
"""

BT2020_COLOURSPACE = RGB_Colourspace(
    'ITU-R BT.2020',
    BT2020_PRIMARIES,
    BT2020_WHITEPOINT,
    BT2020_ILLUMINANT,
    BT2020_TO_XYZ_MATRIX,
    XYZ_TO_BT2020_MATRIX,
    oetf_BT2020,
    eotf_BT2020, )
BT2020_COLOURSPACE.__doc__ = """
*ITU-R BT.2020* colourspace.

References
----------
-   :cite:`InternationalTelecommunicationUnion2015h`

BT2020_COLOURSPACE : RGB_Colourspace
"""
