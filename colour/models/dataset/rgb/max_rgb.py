# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**max_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Max RGB* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np
import colour.models.rgb.derivation

import colour.colorimetry.dataset.illuminants.chromaticity_coordinates
from colour.models.rgb.rgb_colourspace import RGB_Colourspace


__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["MAX_RGB_PRIMARIES",
           "MAX_RGB_WHITEPOINT",
           "MAX_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_MAX_RGB_MATRIX",
           "MAX_RGB_TRANSFER_FUNCTION",
           "MAX_RGB_INVERSE_TRANSFER_FUNCTION",
           "MAX_RGB_COLOURSPACE"]


# http://www.hutchcolor.com/profiles/MaxRGB.zip
MAX_RGB_PRIMARIES = np.array([0.73413379, 0.26586621,
                                 0.10039113, 0.89960887,
                                 0.03621495, 0.]).reshape((3, 2))

MAX_RGB_WHITEPOINT = colour.colorimetry.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

MAX_RGB_TO_XYZ_MATRIX = colour.models.rgb.derivation.get_normalised_primary_matrix(MAX_RGB_PRIMARIES,
                                                                                                     MAX_RGB_WHITEPOINT)

XYZ_TO_MAX_RGB_MATRIX = np.linalg.inv(MAX_RGB_TO_XYZ_MATRIX)

MAX_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

MAX_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

MAX_RGB_COLOURSPACE = RGB_Colourspace("Max RGB",
                                      MAX_RGB_PRIMARIES,
                                      MAX_RGB_WHITEPOINT,
                                      MAX_RGB_TO_XYZ_MATRIX,
                                      XYZ_TO_MAX_RGB_MATRIX,
                                      MAX_RGB_TRANSFER_FUNCTION,
                                      MAX_RGB_INVERSE_TRANSFER_FUNCTION)
