#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**aces_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *ACES RGB* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.exceptions
import color.illuminants
import color.verbose
from color.colorspaces.colorspace import Colorspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "ACES_RGB_PRIMARIES",
           "ACES_RGB_WHITEPOINT",
           "ACES_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_ACES_RGB_MATRIX",
           "ACES_RGB_TRANSFER_FUNCTION",
           "ACES_RGB_INVERSE_TRANSFER_FUNCTION",
           "ACES_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://www.oscars.org/science-technology/council/projects/aces.html
# https://www.dropbox.com/sh/iwd09buudm3lfod/gyjDF-k7oC/ACES_v1.0.1.pdf: 4.1.2 Color space chromaticities
ACES_RGB_PRIMARIES = numpy.matrix([0.73470, 0.26530,
                                   0.00000, 1.00000,
                                   0.00010, -0.07700]).reshape((3, 2))

ACES_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D60")

# https://www.dropbox.com/sh/iwd09buudm3lfod/gyjDF-k7oC/ACES_v1.0.1.pdf: 4.1.4 Converting ACES RGB values to CIE XYZ values
ACES_RGB_TO_XYZ_MATRIX = numpy.matrix([9.52552396e-01, 0.00000000e+00, 9.36786317e-05,
                                       3.43966450e-01, 7.28166097e-01, -7.21325464e-02,
                                       0.00000000e+00, 0.00000000e+00, 1.00882518e+00]).reshape((3, 3))

XYZ_TO_ACES_RGB_MATRIX = ACES_RGB_TO_XYZ_MATRIX.getI()

ACES_RGB_TRANSFER_FUNCTION = lambda x: x

ACES_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x

ACES_RGB_COLORSPACE = Colorspace("ACES RGB",
                                 ACES_RGB_PRIMARIES,
                                 ACES_RGB_WHITEPOINT,
                                 ACES_RGB_TO_XYZ_MATRIX,
                                 XYZ_TO_ACES_RGB_MATRIX,
                                 ACES_RGB_TRANSFER_FUNCTION,
                                 ACES_RGB_INVERSE_TRANSFER_FUNCTION)
