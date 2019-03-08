# -*- coding: utf-8 -*-
"""
FilmLight E-Gamut Colourspace
=============================

Defines the *FilmLight E-Gamut* colourspace:

-   :attr:`colour.models.FILMLIGHT_E_GAMUT_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Siragusano2018a` : Siragusano, D. (2018). Private Discussion with
    Shaw, Nick.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_encoding_FilmLightTLog,
                               log_decoding_FilmLightTLog,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'FILMLIGHT_E_GAMUT_PRIMARIES', 'FILMLIGHT_E_GAMUT_WHITEPOINT_NAME',
    'FILMLIGHT_E_GAMUT_WHITEPOINT', 'FILMLIGHT_E_GAMUT_TO_XYZ_MATRIX',
    'XYZ_TO_FILMLIGHT_E_GAMUT_MATRIX', 'FILMLIGHT_E_GAMUT_COLOURSPACE'
]

FILMLIGHT_E_GAMUT_PRIMARIES = np.array([
    [0.8000, 0.3177],
    [0.1800, 0.9000],
    [0.0650, -0.0805],
])
"""
*FilmLight E-Gamut* colourspace primaries.

FILMLIGHT_E_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

FILMLIGHT_E_GAMUT_WHITEPOINT_NAME = 'D65'
"""
*FilmLight E-Gamut* colourspace whitepoint name.

FILMLIGHT_E_GAMUT_WHITEPOINT : unicode
"""

FILMLIGHT_E_GAMUT_WHITEPOINT = (ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][FILMLIGHT_E_GAMUT_WHITEPOINT_NAME])
"""
*FilmLight E-Gamut* colourspace whitepoint.

FILMLIGHT_E_GAMUT_WHITEPOINT : ndarray
"""

FILMLIGHT_E_GAMUT_TO_XYZ_MATRIX = (normalised_primary_matrix(
    FILMLIGHT_E_GAMUT_PRIMARIES, FILMLIGHT_E_GAMUT_WHITEPOINT))
"""
*FilmLight E-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

FILMLIGHT_E_GAMUT_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_FILMLIGHT_E_GAMUT_MATRIX = (
    np.linalg.inv(FILMLIGHT_E_GAMUT_TO_XYZ_MATRIX))
"""
*CIE XYZ* tristimulus values to *FilmLight E-Gamut* colourspace matrix.

XYZ_TO_FILMLIGHT_E_GAMUT_MATRIX : array_like, (3, 3)
"""

FILMLIGHT_E_GAMUT_COLOURSPACE = RGB_Colourspace(
    'FilmLight E-Gamut',
    FILMLIGHT_E_GAMUT_PRIMARIES,
    FILMLIGHT_E_GAMUT_WHITEPOINT,
    FILMLIGHT_E_GAMUT_WHITEPOINT_NAME,
    FILMLIGHT_E_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_FILMLIGHT_E_GAMUT_MATRIX,
    log_encoding_FilmLightTLog,
    log_decoding_FilmLightTLog,
)
FILMLIGHT_E_GAMUT_COLOURSPACE.__doc__ = """
*FilmLight E-Gamut* colourspace.

    References
    ----------
    :cite:`Siragusano2018a`

FILMLIGHT_E_GAMUT_COLOURSPACE : RGB_Colourspace
"""
