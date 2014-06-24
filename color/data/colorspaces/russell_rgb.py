# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**russell_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *Russell RGB* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.data.illuminants.chromaticity_coordinates
import color.computation.derivation
import color.utilities.exceptions
import color.utilities.verbose
from color.computation.colorspace import Colorspace


__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["RUSSELL_RGB_PRIMARIES",
           "RUSSELL_RGB_WHITEPOINT",
           "RUSSELL_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_RUSSELL_RGB_MATRIX",
           "RUSSELL_RGB_TRANSFER_FUNCTION",
           "RUSSELL_RGB_INVERSE_TRANSFER_FUNCTION",
           "RUSSELL_RGB_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://www.russellcottrell.com/photo/RussellRGB.htm
RUSSELL_RGB_PRIMARIES = numpy.matrix([0.6900, 0.3100,
                                      0.1800, 0.7700,
                                      0.1000, 0.0200]).reshape((3, 2))

RUSSELL_RGB_WHITEPOINT = color.data.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D55")

RUSSELL_RGB_TO_XYZ_MATRIX = color.computation.derivation.get_normalized_primary_matrix(RUSSELL_RGB_PRIMARIES,
                                                                                        RUSSELL_RGB_WHITEPOINT)

XYZ_TO_RUSSELL_RGB_MATRIX = RUSSELL_RGB_TO_XYZ_MATRIX.getI()

RUSSELL_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

RUSSELL_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

RUSSELL_RGB_COLORSPACE = Colorspace("Russell RGB",
                                    RUSSELL_RGB_PRIMARIES,
                                    RUSSELL_RGB_WHITEPOINT,
                                    RUSSELL_RGB_TO_XYZ_MATRIX,
                                    XYZ_TO_RUSSELL_RGB_MATRIX,
                                    RUSSELL_RGB_TRANSFER_FUNCTION,
                                    RUSSELL_RGB_INVERSE_TRANSFER_FUNCTION)
