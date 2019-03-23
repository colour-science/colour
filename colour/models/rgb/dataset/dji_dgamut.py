# -*- coding: utf-8 -*-
"""
DJI D-Gamut Colourspace
=======================

Defines the *DJI D-Gamut* colourspace:

-   :attr:`colour.models.DJI_D_GAMUT_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`DJI2017` : Dji. (2017). White Paper on D-Log and D-Gamut of DJI
    Cinema Color System. Retrieved from https://dl.djicdn.com/downloads/\
zenmuse+x7/20171010/D-Log_D-Gamut_Whitepaper.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_encoding_DJIDLog,
                               log_decoding_DJIDLog)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'DJI_D_GAMUT_PRIMARIES', 'DJI_D_GAMUT_WHITEPOINT_NAME',
    'DJI_D_GAMUT_WHITEPOINT', 'DJI_D_GAMUT_TO_XYZ_MATRIX',
    'XYZ_TO_DJI_D_GAMUT_MATRIX', 'DJI_D_GAMUT_COLOURSPACE'
]

DJI_D_GAMUT_PRIMARIES = np.array([
    [0.71, 0.31],
    [0.21, 0.88],
    [0.09, -0.08],
])
"""
*DJI D-Gamut* colourspace primaries.

DJI_D_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

DJI_D_GAMUT_WHITEPOINT_NAME = 'D65'
"""
*DJI D-Gamut* colourspace whitepoint name.

DJI_D_GAMUT_WHITEPOINT : unicode
"""

DJI_D_GAMUT_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    DJI_D_GAMUT_WHITEPOINT_NAME])
"""
*DJI D-Gamut* colourspace whitepoint.

DJI_D_GAMUT_WHITEPOINT : ndarray
"""

DJI_D_GAMUT_TO_XYZ_MATRIX = np.array([[0.6482, 0.1940,
                                       0.1082], [0.2830, 0.8132, -0.0962],
                                      [-0.0183, -0.0832, 1.1903]])
"""
*DJI D-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

DJI_D_GAMUT_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_DJI_D_GAMUT_MATRIX = np.array([[1.7257, -0.4314,
                                       -0.1917], [-0.6025, 1.3906, 0.1671],
                                      [-0.0156, 0.0905, 0.8489]])
"""
*CIE XYZ* tristimulus values to *DJI D-Gamut* colourspace matrix.

XYZ_TO_DJI_D_GAMUT_MATRIX : array_like, (3, 3)
"""

DJI_D_GAMUT_COLOURSPACE = RGB_Colourspace(
    'DJI D-Gamut',
    DJI_D_GAMUT_PRIMARIES,
    DJI_D_GAMUT_WHITEPOINT,
    DJI_D_GAMUT_WHITEPOINT_NAME,
    DJI_D_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_DJI_D_GAMUT_MATRIX,
    log_encoding_DJIDLog,
    log_decoding_DJIDLog,
)
DJI_D_GAMUT_COLOURSPACE.__doc__ = """
*DJI_D-Gamut* colourspace.

    References
    ----------
    :cite:`DJI2017`

DJI_D_GAMUT_COLOURSPACE : RGB_Colourspace
"""
