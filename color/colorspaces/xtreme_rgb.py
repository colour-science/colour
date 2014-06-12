#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**xtreme_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *Xtreme RGB* colorspace.

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
           "XTREME_RGB_PRIMARIES",
           "XTREME_RGB_WHITEPOINT",
           "XTREME_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_XTREME_RGB_MATRIX",
           "XTREME_RGB_TRANSFER_FUNCTION",
           "XTREME_RGB_INVERSE_TRANSFER_FUNCTION",
           "XTREME_RGB_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://www.hutchcolor.com/profiles/XtremeRGB.zip
XTREME_RGB_PRIMARIES = numpy.matrix([1., 0.,
                                     0., 1.,
                                     0., 0.]).reshape((3, 2))

XTREME_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("CIE 1931 2 Degree Standard Observer").get("D50")

XTREME_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(XTREME_RGB_PRIMARIES, XTREME_RGB_WHITEPOINT)

XYZ_TO_XTREME_RGB_MATRIX = XTREME_RGB_TO_XYZ_MATRIX.getI()

XTREME_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

XTREME_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

XTREME_RGB_COLORSPACE = Colorspace("Xtreme RGB",
                                   XTREME_RGB_PRIMARIES,
                                   XTREME_RGB_WHITEPOINT,
                                   XTREME_RGB_TO_XYZ_MATRIX,
                                   XYZ_TO_XTREME_RGB_MATRIX,
                                   XTREME_RGB_TRANSFER_FUNCTION,
                                   XTREME_RGB_INVERSE_TRANSFER_FUNCTION)
