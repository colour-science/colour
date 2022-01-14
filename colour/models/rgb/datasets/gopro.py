# -*- coding: utf-8 -*-
"""
GoPro Colourspaces
==================

Defines the *GoPro* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_PROTUNE_NATIVE`.

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

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    log_decoding_Protune,
    log_encoding_Protune,
    normalised_primary_matrix,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PRIMARIES_PROTUNE_NATIVE',
    'WHITEPOINT_NAME_PROTUNE_NATIVE',
    'CCS_WHITEPOINT_PROTUNE_NATIVE',
    'MATRIX_PROTUNE_NATIVE_TO_XYZ',
    'MATRIX_XYZ_TO_PROTUNE_NATIVE',
    'RGB_COLOURSPACE_PROTUNE_NATIVE',
]

PRIMARIES_PROTUNE_NATIVE: NDArray = np.array([
    [0.698480461493841, 0.193026445370121],
    [0.329555378387345, 1.024596624134644],
    [0.108442631407675, -0.034678569754016],
])
"""
*Protune Native* colourspace primaries.
"""

WHITEPOINT_NAME_PROTUNE_NATIVE: str = 'D65'
"""
*Protune Native* colourspace whitepoint name.
"""

CCS_WHITEPOINT_PROTUNE_NATIVE: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_PROTUNE_NATIVE])
"""
*Protune Native* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_PROTUNE_NATIVE_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_PROTUNE_NATIVE, CCS_WHITEPOINT_PROTUNE_NATIVE)
"""
*Protune Native* colourspace to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_PROTUNE_NATIVE: NDArray = np.linalg.inv(
    MATRIX_PROTUNE_NATIVE_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *Protune Native* colourspace matrix.
"""

RGB_COLOURSPACE_PROTUNE_NATIVE: RGB_Colourspace = RGB_Colourspace(
    'Protune Native',
    PRIMARIES_PROTUNE_NATIVE,
    CCS_WHITEPOINT_PROTUNE_NATIVE,
    WHITEPOINT_NAME_PROTUNE_NATIVE,
    MATRIX_PROTUNE_NATIVE_TO_XYZ,
    MATRIX_XYZ_TO_PROTUNE_NATIVE,
    log_encoding_Protune,
    log_decoding_Protune,
)
RGB_COLOURSPACE_PROTUNE_NATIVE.__doc__ = """
*Protune Native* colourspace.

References
----------
:cite:`GoPro2016a`, :cite:`Mansencal2015d`
"""
