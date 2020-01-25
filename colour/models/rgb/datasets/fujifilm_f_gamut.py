# -*- coding: utf-8 -*-
"""
Fujifilm F-Gamut Colourspace
============================

Defines the *Fujifilm F-Gamut* colourspace:

-   :attr:`colour.models.F_GAMUT_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Fujifilm2016` : Fujifilm. (2016). F-Log Data Sheet Ver.1.0. \
Retrieved from https://www.fujifilm.com/support/digital_cameras/\
software/lut/pdf/F-Log_DataSheet_E_Ver.1.0.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_encoding_FLog,
                               normalised_primary_matrix, log_decoding_FLog)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'F_GAMUT_PRIMARIES', 'F_GAMUT_WHITEPOINT_NAME', 'F_GAMUT_WHITEPOINT',
    'F_GAMUT_TO_XYZ_MATRIX', 'XYZ_TO_F_GAMUT_MATRIX', 'F_GAMUT_COLOURSPACE'
]

F_GAMUT_PRIMARIES = np.array([
    [0.70800, 0.29200],
    [0.17000, 0.79700],
    [0.13100, 0.04600],
])
"""
*Fujifilm F-Gamut* colourspace primaries.

F_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

F_GAMUT_WHITEPOINT_NAME = 'D65'
"""
*Fujifilm F-Gamut* colourspace whitepoint name.

F_GAMUT_WHITEPOINT : unicode
"""

F_GAMUT_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    F_GAMUT_WHITEPOINT_NAME])
"""
*Fujifilm F-Gamut* colourspace whitepoint.

F_GAMUT_WHITEPOINT : ndarray
"""

F_GAMUT_TO_XYZ_MATRIX = normalised_primary_matrix(F_GAMUT_PRIMARIES,
                                                  F_GAMUT_WHITEPOINT)
"""
*Fujifilm F-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

F_GAMUT_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_F_GAMUT_MATRIX = np.linalg.inv(F_GAMUT_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Fujifilm F-Gamut* colourspace matrix.

XYZ_TO_F_GAMUT_MATRIX : array_like, (3, 3)
"""

F_GAMUT_COLOURSPACE = RGB_Colourspace(
    'F-Gamut',
    F_GAMUT_PRIMARIES,
    F_GAMUT_WHITEPOINT,
    F_GAMUT_WHITEPOINT_NAME,
    F_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_F_GAMUT_MATRIX,
    log_encoding_FLog,
    log_decoding_FLog,
)
F_GAMUT_COLOURSPACE.__doc__ = """
*Fujifilm F-Gamut* colourspace.

References
----------
:cite:`Fujifilm2016`

F_GAMUT_COLOURSPACE : RGB_Colourspace
"""
