# -*- coding: utf-8 -*-
"""
Sharp RGB Colourspace
=====================

Defines the *Sharp RGB* colourspace:

-   :attr:`colour.models.SHARP_RGB_COLOURSPACE`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Susstrunk2000` : Susstrunk, S. E., Holm, J. M., & Finlayson, G. D.
    (2000). Chromatic adaptation performance of different RGB sensors.
    In R. Eschbach & G. G. Marcu (Eds.), Photonics West 2001 - Electronic
    Imaging (Vol. 4300, pp. 172-183). doi:10.1117/12.410788
-   :cite:`Ward2002` : Ward, G., & Eydelberg-Vileshin, E. (2002). Picture
    Perfect RGB Rendering Using Spectral Prefiltering and Sharp Color
    Primaries. Eurographics Workshop on Rendering, 117-124.
    doi:10.2312/EGWR/EGWR02/117-124
-   :cite:`Ward2016` : Ward, G. (2016). Private Discussion with Mansencal, T.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, linear_function,
                               normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SHARP_RGB_PRIMARIES', 'SHARP_RGB_WHITEPOINT_NAME', 'SHARP_RGB_WHITEPOINT',
    'SHARP_RGB_TO_XYZ_MATRIX', 'XYZ_TO_SHARP_RGB_MATRIX',
    'SHARP_RGB_COLOURSPACE'
]

SHARP_RGB_PRIMARIES = np.array([
    [0.6898, 0.3206],
    [0.0736, 0.9003],
    [0.1166, 0.0374],
])
"""
*Sharp RGB* colourspace primaries.

Notes
-----
The primaries were originally derived from the :math:`M_{Sharp}` matrix as
given in *Ward and Eydelberg-Vileshin (2002)*:

    M_Sharp = np.array(
        [[1.2694, -0.0988, -0.1706],
         [-0.8364, 1.8006, 0.0357],
         [0.0297, -0.0315, 1.0018]])

    P, W = (
        array([[ 0.68976058,  0.32060751],
               [ 0.07358274,  0.90029055],
               [ 0.1166078 ,  0.0373923 ]]),
        array([ 0.33332778,  0.33334544]))

Private discussion with Ward (2016) confirmed he used the following primaries
and whitepoint:

    [0.6898, 0.3206, 0.0736, 0.9003, 0.1166, 0.0374, 1 / 3, 1 / 3]

SHARP_RGB_PRIMARIES : ndarray, (3, 2)
"""

SHARP_RGB_WHITEPOINT_NAME = 'E'
"""
*Sharp RGB* colourspace whitepoint name.

SHARP_RGB_WHITEPOINT_NAME : unicode
"""

SHARP_RGB_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    SHARP_RGB_WHITEPOINT_NAME])
"""
*Sharp RGB* colourspace whitepoint.

SHARP_RGB_WHITEPOINT : ndarray
"""

SHARP_RGB_TO_XYZ_MATRIX = normalised_primary_matrix(SHARP_RGB_PRIMARIES,
                                                    SHARP_RGB_WHITEPOINT)
"""
*Sharp RGB* colourspace to *CIE XYZ* tristimulus values matrix.

SHARP_RGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_SHARP_RGB_MATRIX = np.linalg.inv(SHARP_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Sharp RGB* colourspace matrix.

XYZ_TO_SHARP_RGB_MATRIX : array_like, (3, 3)
"""

SHARP_RGB_COLOURSPACE = RGB_Colourspace(
    'Sharp RGB',
    SHARP_RGB_PRIMARIES,
    SHARP_RGB_WHITEPOINT,
    SHARP_RGB_WHITEPOINT_NAME,
    SHARP_RGB_TO_XYZ_MATRIX,
    XYZ_TO_SHARP_RGB_MATRIX,
    linear_function,
    linear_function,
)
SHARP_RGB_COLOURSPACE.__doc__ = """
*Sharp RGB* colourspace.

References
----------
:cite:`Susstrunk2000`, :cite:`Ward2002`, :cite:`Ward2016`

SHARP_RGB_COLOURSPACE : RGB_Colourspace
"""
