#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**adobe_wide_gamut_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *Adobe Wide Gamut RGB* colorspace.

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
           "ADOBE_WIDE_GAMUT_RGB_PRIMARIES",
           "ADOBE_WIDE_GAMUT_RGB_WHITEPOINT",
           "ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX",
           "ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION",
           "ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION",
           "ADOBE_WIDE_GAMUT_RGB_COLORSPACE"]

LOGGER = color.utilities.verbose.install_logger()

# http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space
ADOBE_WIDE_GAMUT_RGB_PRIMARIES = numpy.matrix([0.7347, 0.2653,
                                               0.1152, 0.8264,
                                               0.1566, 0.0177]).reshape((3, 2))

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
                                                                                    ADOBE_WIDE_GAMUT_RGB_WHITEPOINT)

XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX = ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX.getI()

ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / (563. / 256.))

ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** (563. / 256.)

ADOBE_WIDE_GAMUT_RGB_COLORSPACE = Colorspace("Adobe Wide Gamut RGB",
                                             ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
                                             ADOBE_WIDE_GAMUT_RGB_WHITEPOINT,
                                             ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
                                             XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX,
                                             ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION,
                                             ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION)
