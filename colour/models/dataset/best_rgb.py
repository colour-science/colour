#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**best_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Best RGB* colourspace.

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

__all__ = ["BEST_RGB_PRIMARIES",
           "BEST_RGB_WHITEPOINT",
           "BEST_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_BEST_RGB_MATRIX",
           "BEST_RGB_TRANSFER_FUNCTION",
           "BEST_RGB_INVERSE_TRANSFER_FUNCTION",
           "BEST_RGB_COLOURSPACE"]

# http://www.hutchcolor.com/profiles/BestRGB.zip
BEST_RGB_PRIMARIES = np.array(
    [0.73519163763066209, 0.26480836236933797,
     0.2153361344537815, 0.77415966386554624,
     0.13012295081967212, 0.034836065573770496]).reshape((3, 2))

BEST_RGB_WHITEPOINT = ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

BEST_RGB_TO_XYZ_MATRIX = get_normalised_primary_matrix(BEST_RGB_PRIMARIES,
                                                       BEST_RGB_WHITEPOINT)

XYZ_TO_BEST_RGB_MATRIX = np.linalg.inv(BEST_RGB_TO_XYZ_MATRIX)

BEST_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

BEST_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

BEST_RGB_COLOURSPACE = RGB_Colourspace(
    "Best RGB",
    BEST_RGB_PRIMARIES,
    BEST_RGB_WHITEPOINT,
    BEST_RGB_TO_XYZ_MATRIX,
    XYZ_TO_BEST_RGB_MATRIX,
    BEST_RGB_TRANSFER_FUNCTION,
    BEST_RGB_INVERSE_TRANSFER_FUNCTION)
