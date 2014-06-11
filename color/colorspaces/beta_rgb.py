#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**beta_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *Beta RGB* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.derivation
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
           "BETA_RGB_PRIMARIES",
           "BETA_RGB_WHITEPOINT",
           "BETA_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_BETA_RGB_MATRIX",
           "BETA_RGB_TRANSFER_FUNCTION",
           "BETA_RGB_INVERSE_TRANSFER_FUNCTION",
           "BETA_RGB_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://www.brucelindbloom.com/WorkingSpaceInfo.html
BETA_RGB_PRIMARIES = numpy.matrix([0.6888, 0.3112,
                                   0.1986, 0.7551,
                                   0.1265, 0.0352]).reshape((3, 2))

BETA_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

BETA_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(BETA_RGB_PRIMARIES, BETA_RGB_WHITEPOINT)

XYZ_TO_BETA_RGB_MATRIX = BETA_RGB_TO_XYZ_MATRIX.getI()

BETA_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

BETA_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

BETA_RGB_COLORSPACE = Colorspace("Beta RGB",
                                 BETA_RGB_PRIMARIES,
                                 BETA_RGB_WHITEPOINT,
                                 BETA_RGB_TO_XYZ_MATRIX,
                                 XYZ_TO_BETA_RGB_MATRIX,
                                 BETA_RGB_TRANSFER_FUNCTION,
                                 BETA_RGB_INVERSE_TRANSFER_FUNCTION)
