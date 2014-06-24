# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**apple_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *Apple RGB* colorspace.

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

__all__ = ["APPLE_RGB_PRIMARIES",
           "APPLE_RGB_WHITEPOINT",
           "APPLE_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_APPLE_RGB_MATRIX",
           "APPLE_RGB_TRANSFER_FUNCTION",
           "APPLE_RGB_INVERSE_TRANSFER_FUNCTION",
           "APPLE_RGB_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://www.brucelindbloom.com/WorkingSpaceInfo.html
APPLE_RGB_PRIMARIES = numpy.matrix([0.6250, 0.3400,
                                    0.2800, 0.5950,
                                    0.1550, 0.0700]).reshape((3, 2))

APPLE_RGB_WHITEPOINT = color.data.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D65")

APPLE_RGB_TO_XYZ_MATRIX = color.computation.derivation.get_normalized_primary_matrix(APPLE_RGB_PRIMARIES,
                                                                                      APPLE_RGB_WHITEPOINT)

XYZ_TO_APPLE_RGB_MATRIX = APPLE_RGB_TO_XYZ_MATRIX.getI()

APPLE_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 1.8)

APPLE_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 1.8

APPLE_RGB_COLORSPACE = Colorspace("Apple RGB",
                                  APPLE_RGB_PRIMARIES,
                                  APPLE_RGB_WHITEPOINT,
                                  APPLE_RGB_TO_XYZ_MATRIX,
                                  XYZ_TO_APPLE_RGB_MATRIX,
                                  APPLE_RGB_TRANSFER_FUNCTION,
                                  APPLE_RGB_INVERSE_TRANSFER_FUNCTION)
