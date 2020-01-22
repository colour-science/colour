# -*- coding: utf-8 -*-
"""
GoPro Colourspaces
==================

Defines the *GoPro* colourspaces:

-   :attr:`colour.models.PROTUNE_NATIVE_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

Notes
-----
-   The *Protune Native* colourspace primaries were derived using the method
    outlined in :cite:`Mansencal2015d` followed with a chromatic adaptation
    step to *CIE Standard Illuminant D Series D65* using
    :func:`colour.chromatically_adapted_primaries` definition.

References
----------
-   :cite:`GoPro2016a` : GoPro, Duiker, H.-P., & Mansencal, T. (2016).
    gopro.py. Retrieved April 12, 2017, from
    https://github.com/hpd/OpenColorIO-Configs/blob/master/aces_1.0.3/python/\
aces_ocio/colorspaces/gopro.py
-   :cite:`Mansencal2015d` : Mansencal, T. (2015). RED Colourspaces Derivation.
    Retrieved May 20, 2015, from
    https://www.colour-science.org/posts/red-colourspaces-derivation
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_decoding_Protune,
                               log_encoding_Protune, normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PROTUNE_NATIVE_PRIMARIES', 'PROTUNE_NATIVE_WHITEPOINT_NAME',
    'PROTUNE_NATIVE_WHITEPOINT', 'PROTUNE_NATIVE_TO_XYZ_MATRIX',
    'XYZ_TO_PROTUNE_NATIVE_MATRIX', 'PROTUNE_NATIVE_COLOURSPACE'
]

PROTUNE_NATIVE_PRIMARIES = np.array([
    [0.698480461493841, 0.193026445370121],
    [0.329555378387345, 1.024596624134644],
    [0.108442631407675, -0.034678569754016],
])
"""
*Protune Native* colourspace primaries.

PROTUNE_NATIVE_PRIMARIES : ndarray, (3, 2)
"""

PROTUNE_NATIVE_WHITEPOINT_NAME = 'D65'
"""
*Protune Native* colourspace whitepoint name.

PROTUNE_NATIVE_WHITEPOINT_NAME : unicode
"""

PROTUNE_NATIVE_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
                             [PROTUNE_NATIVE_WHITEPOINT_NAME])
"""
*Protune Native* colourspace whitepoint.

PROTUNE_NATIVE_WHITEPOINT : ndarray
"""

PROTUNE_NATIVE_TO_XYZ_MATRIX = normalised_primary_matrix(
    PROTUNE_NATIVE_PRIMARIES, PROTUNE_NATIVE_WHITEPOINT)
"""
*Protune Native* colourspace to *CIE XYZ* tristimulus values matrix.

PROTUNE_NATIVE_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_PROTUNE_NATIVE_MATRIX = np.linalg.inv(PROTUNE_NATIVE_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Protune Native* colourspace matrix.

XYZ_TO_PROTUNE_NATIVE_MATRIX : array_like, (3, 3)
"""

PROTUNE_NATIVE_COLOURSPACE = RGB_Colourspace(
    'Protune Native',
    PROTUNE_NATIVE_PRIMARIES,
    PROTUNE_NATIVE_WHITEPOINT,
    PROTUNE_NATIVE_WHITEPOINT_NAME,
    PROTUNE_NATIVE_TO_XYZ_MATRIX,
    XYZ_TO_PROTUNE_NATIVE_MATRIX,
    log_encoding_Protune,
    log_decoding_Protune,
)
PROTUNE_NATIVE_COLOURSPACE.__doc__ = """
*Protune Native* colourspace.

References
----------
:cite:`GoPro2016a`, :cite:`Mansencal2015d`

PROTUNE_NATIVE_COLOURSPACE : RGB_Colourspace
"""
