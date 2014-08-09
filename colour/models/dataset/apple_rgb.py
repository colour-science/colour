#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**apple_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Apple RGB* colourspace.

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

__all__ = ["APPLE_RGB_PRIMARIES",
           "APPLE_RGB_WHITEPOINT",
           "APPLE_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_APPLE_RGB_MATRIX",
           "APPLE_RGB_TRANSFER_FUNCTION",
           "APPLE_RGB_INVERSE_TRANSFER_FUNCTION",
           "APPLE_RGB_COLOURSPACE"]

# http://www.brucelindbloom.com/WorkingSpaceInfo.html
APPLE_RGB_PRIMARIES = np.array(
    [0.6250, 0.3400,
     0.2800, 0.5950,
     0.1550, 0.0700]).reshape((3, 2))

APPLE_RGB_WHITEPOINT = ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D65")

APPLE_RGB_TO_XYZ_MATRIX = get_normalised_primary_matrix(APPLE_RGB_PRIMARIES,
                                                        APPLE_RGB_WHITEPOINT)

XYZ_TO_APPLE_RGB_MATRIX = np.linalg.inv(APPLE_RGB_TO_XYZ_MATRIX)

APPLE_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 1.8)

APPLE_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 1.8

APPLE_RGB_COLOURSPACE = RGB_Colourspace(
    "Apple RGB",
    APPLE_RGB_PRIMARIES,
    APPLE_RGB_WHITEPOINT,
    APPLE_RGB_TO_XYZ_MATRIX,
    XYZ_TO_APPLE_RGB_MATRIX,
    APPLE_RGB_TRANSFER_FUNCTION,
    APPLE_RGB_INVERSE_TRANSFER_FUNCTION)
