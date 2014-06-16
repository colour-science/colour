#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cie_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *CIE RGB* colorspace.

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

__all__ = ["CIE_RGB_PRIMARIES",
           "CIE_RGB_WHITEPOINT",
           "CIE_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_CIE_RGB_MATRIX",
           "CIE_RGB_TRANSFER_FUNCTION",
           "CIE_RGB_INVERSE_TRANSFER_FUNCTION",
           "CIE_RGB_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright.E2.80.93Guild_data
CIE_RGB_PRIMARIES = numpy.matrix([0.7350, 0.2650,
                                  0.2740, 0.7170,
                                  0.1670, 0.0090]).reshape((3, 2))

CIE_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("CIE 1931 2 Degree Standard Observer").get("E")

CIE_RGB_TO_XYZ_MATRIX = 1. / 0.17697 * numpy.matrix([0.49, 0.31, 0.20,
                                                    0.17697, 0.81240, 0.01063,
                                                    0.00, 0.01, 0.99]).reshape((3, 3))

XYZ_TO_CIE_RGB_MATRIX = CIE_RGB_TO_XYZ_MATRIX.getI()

CIE_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

CIE_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

CIE_RGB_COLORSPACE = Colorspace("CIE RGB",
                                CIE_RGB_PRIMARIES,
                                CIE_RGB_WHITEPOINT,
                                CIE_RGB_TO_XYZ_MATRIX,
                                XYZ_TO_CIE_RGB_MATRIX,
                                CIE_RGB_TRANSFER_FUNCTION,
                                CIE_RGB_INVERSE_TRANSFER_FUNCTION)
