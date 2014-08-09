#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**adobe_wide_gamut_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Adobe Wide Gamut RGB* colourspace.

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

__all__ = ["ADOBE_WIDE_GAMUT_RGB_PRIMARIES",
           "ADOBE_WIDE_GAMUT_RGB_WHITEPOINT",
           "ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX",
           "ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION",
           "ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION",
           "ADOBE_WIDE_GAMUT_RGB_COLOURSPACE"]

# http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space
ADOBE_WIDE_GAMUT_RGB_PRIMARIES = np.array(
    [0.7347, 0.2653,
     0.1152, 0.8264,
     0.1566, 0.0177]).reshape((3, 2))

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT = ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = get_normalised_primary_matrix(
    ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
    ADOBE_WIDE_GAMUT_RGB_WHITEPOINT)

XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX = np.linalg.inv(
    ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX)

ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / (563. / 256.))

ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** (563. / 256.)

ADOBE_WIDE_GAMUT_RGB_COLOURSPACE = RGB_Colourspace(
    "Adobe Wide Gamut RGB",
    ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
    ADOBE_WIDE_GAMUT_RGB_WHITEPOINT,
    ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX,
    ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION,
    ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION)
