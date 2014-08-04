# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**xtreme_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Xtreme RGB* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, get_normalised_primary_matrix

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["XTREME_RGB_PRIMARIES",
           "XTREME_RGB_WHITEPOINT",
           "XTREME_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_XTREME_RGB_MATRIX",
           "XTREME_RGB_TRANSFER_FUNCTION",
           "XTREME_RGB_INVERSE_TRANSFER_FUNCTION",
           "XTREME_RGB_COLOURSPACE"]


# http://www.hutchcolor.com/profiles/XtremeRGB.zip
XTREME_RGB_PRIMARIES = np.array(
    [1., 0.,
     0., 1.,
     0., 0.]).reshape((3, 2))

XTREME_RGB_WHITEPOINT = ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

XTREME_RGB_TO_XYZ_MATRIX = get_normalised_primary_matrix(XTREME_RGB_PRIMARIES,
                                                         XTREME_RGB_WHITEPOINT)

XYZ_TO_XTREME_RGB_MATRIX = np.linalg.inv(XTREME_RGB_TO_XYZ_MATRIX)

XTREME_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

XTREME_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

XTREME_RGB_COLOURSPACE = RGB_Colourspace(
    "Xtreme RGB",
    XTREME_RGB_PRIMARIES,
    XTREME_RGB_WHITEPOINT,
    XTREME_RGB_TO_XYZ_MATRIX,
    XYZ_TO_XTREME_RGB_MATRIX,
    XTREME_RGB_TRANSFER_FUNCTION,
    XTREME_RGB_INVERSE_TRANSFER_FUNCTION)
