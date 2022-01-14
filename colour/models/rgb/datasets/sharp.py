# -*- coding: utf-8 -*-
"""
Sharp RGB Colourspace
=====================

Defines the *Sharp RGB* colourspace:

-   :attr:`colour.models.RGB_COLOURSPACE_SHARP_RGB`

References
----------
-   :cite:`Susstrunk2000` : Susstrunk, S. E., Holm, J. M., & Finlayson, G. D.
    (2000). Chromatic adaptation performance of different RGB sensors. In R.
    Eschbach & G. G. Marcu (Eds.), Photonics West 2001 - Electronic Imaging
    (Vol. 4300, Issue January, pp. 172-183). doi:10.1117/12.410788
-   :cite:`Ward2002` : Ward, G., & Eydelberg-Vileshin, E. (2002). Picture
    Perfect RGB Rendering Using Spectral Prefiltering and Sharp Color
    Primaries. Eurographics Workshop on Rendering, 117-124.
    doi:10.2312/EGWR/EGWR02/117-124
-   :cite:`Ward2016` : Borer, T. (2017). Private Discussion with Mansencal, T.
    and Shaw, N.
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    linear_function,
    normalised_primary_matrix,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_SHARP_RGB',
    'WHITEPOINT_NAME_SHARP_RGB',
    'CCS_WHITEPOINT_SHARP_RGB',
    'MATRIX_SHARP_RGB_TO_XYZ',
    'MATRIX_XYZ_TO_SHARP_RGB',
    'RGB_COLOURSPACE_SHARP_RGB',
]

PRIMARIES_SHARP_RGB: NDArray = np.array([
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
"""

WHITEPOINT_NAME_SHARP_RGB: str = 'E'
"""
*Sharp RGB* colourspace whitepoint name.
"""

CCS_WHITEPOINT_SHARP_RGB: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_SHARP_RGB])
"""
*Sharp RGB* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_SHARP_RGB_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_SHARP_RGB, CCS_WHITEPOINT_SHARP_RGB)
"""
*Sharp RGB* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_SHARP_RGB: NDArray = np.linalg.inv(MATRIX_SHARP_RGB_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Sharp RGB* colourspace matrix.
"""

RGB_COLOURSPACE_SHARP_RGB: RGB_Colourspace = RGB_Colourspace(
    'Sharp RGB',
    PRIMARIES_SHARP_RGB,
    CCS_WHITEPOINT_SHARP_RGB,
    WHITEPOINT_NAME_SHARP_RGB,
    MATRIX_SHARP_RGB_TO_XYZ,
    MATRIX_XYZ_TO_SHARP_RGB,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_SHARP_RGB.__doc__ = """
*Sharp RGB* colourspace.

References
----------
:cite:`Susstrunk2000`, :cite:`Ward2002`, :cite:`Ward2016`
"""
