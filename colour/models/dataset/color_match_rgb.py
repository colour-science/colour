#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**color_match_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *ColourMatch RGB* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, get_normalised_primary_matrix

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["COLOR_MATCH_RGB_PRIMARIES",
           "COLOR_MATCH_RGB_WHITEPOINT",
           "COLOR_MATCH_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_COLOR_MATCH_RGB_MATRIX",
           "COLOR_MATCH_RGB_TRANSFER_FUNCTION",
           "COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION",
           "COLOR_MATCH_RGB_COLOURSPACE"]

# http://www.brucelindbloom.com/WorkingSpaceInfo.html
COLOR_MATCH_RGB_PRIMARIES = np.array(
    [0.6300, 0.3400,
     0.2950, 0.6050,
     0.1500, 0.0750]).reshape((3, 2))

COLOR_MATCH_RGB_WHITEPOINT = ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

COLOR_MATCH_RGB_TO_XYZ_MATRIX = get_normalised_primary_matrix(
    COLOR_MATCH_RGB_PRIMARIES,
    COLOR_MATCH_RGB_WHITEPOINT)

XYZ_TO_COLOR_MATCH_RGB_MATRIX = np.linalg.inv(COLOR_MATCH_RGB_TO_XYZ_MATRIX)

COLOR_MATCH_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 1.8)

COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 1.8

COLOR_MATCH_RGB_COLOURSPACE = RGB_Colourspace(
    "ColorMatch RGB",
    COLOR_MATCH_RGB_PRIMARIES,
    COLOR_MATCH_RGB_WHITEPOINT,
    COLOR_MATCH_RGB_TO_XYZ_MATRIX,
    XYZ_TO_COLOR_MATCH_RGB_MATRIX,
    COLOR_MATCH_RGB_TRANSFER_FUNCTION,
    COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION)
