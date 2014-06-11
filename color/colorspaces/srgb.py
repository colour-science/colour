#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**srgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *sRGB* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.illuminants
import color.utilities.exceptions
import color.utilities.verbose
from color.colorspaces.colorspace import Colorspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "sRGB_PRIMARIES",
           "sRGB_WHITEPOINT",
           "sRGB_TO_XYZ_MATRIX",
           "XYZ_TO_sRGB_MATRIX",
           "sRGB_TRANSFER_FUNCTION",
           "sRGB_INVERSE_TRANSFER_FUNCTION",
           "sRGB_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://www.color.org/srgb.pdf
# http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-5-200204-I!!PDF-E.pdf: 1 Opto-electronic conversion
sRGB_PRIMARIES = numpy.matrix([0.6400, 0.3300,
                               0.3000, 0.6000,
                               0.1500, 0.0600]).reshape((3, 2))

sRGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D65")

sRGB_TO_XYZ_MATRIX = numpy.matrix([0.41238656, 0.35759149, 0.18045049,
                                   0.21263682, 0.71518298, 0.0721802,
                                   0.01933062, 0.11919716, 0.95037259]).reshape((3, 3))

XYZ_TO_sRGB_MATRIX = sRGB_TO_XYZ_MATRIX.getI()

sRGB_TRANSFER_FUNCTION = lambda x: x * 12.92 if x <= 0.0031308 else 1.055 * (x ** (1 / 2.4)) - 0.055

sRGB_INVERSE_TRANSFER_FUNCTION = lambda x: x / 12.92 if x <= 0.0031308 else ((x + 0.055) / 1.055) ** 2.4

sRGB_COLORSPACE = Colorspace("sRGB",
                             sRGB_PRIMARIES,
                             sRGB_WHITEPOINT,
                             sRGB_TO_XYZ_MATRIX,
                             XYZ_TO_sRGB_MATRIX,
                             sRGB_TRANSFER_FUNCTION,
                             sRGB_INVERSE_TRANSFER_FUNCTION)
