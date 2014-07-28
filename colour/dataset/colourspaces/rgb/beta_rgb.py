# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**beta_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *Beta RGB* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.computation.colourspaces.rgb.derivation
import colour.dataset.illuminants.chromaticity_coordinates
from colour.computation.colourspaces.rgb.rgb_colourspace import RGB_Colourspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["BETA_RGB_PRIMARIES",
           "BETA_RGB_WHITEPOINT",
           "BETA_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_BETA_RGB_MATRIX",
           "BETA_RGB_TRANSFER_FUNCTION",
           "BETA_RGB_INVERSE_TRANSFER_FUNCTION",
           "BETA_RGB_COLOURSPACE"]


# http://www.brucelindbloom.com/WorkingSpaceInfo.html
BETA_RGB_PRIMARIES = numpy.array([0.6888, 0.3112,
                                  0.1986, 0.7551,
                                  0.1265, 0.0352]).reshape((3, 2))

BETA_RGB_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

BETA_RGB_TO_XYZ_MATRIX = colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix(
    BETA_RGB_PRIMARIES,
    BETA_RGB_WHITEPOINT)

XYZ_TO_BETA_RGB_MATRIX = numpy.linalg.inv(BETA_RGB_TO_XYZ_MATRIX)

BETA_RGB_TRANSFER_FUNCTION = lambda x: x ** (1 / 2.2)

BETA_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x ** 2.2

BETA_RGB_COLOURSPACE = RGB_Colourspace("Beta RGB",
                                       BETA_RGB_PRIMARIES,
                                       BETA_RGB_WHITEPOINT,
                                       BETA_RGB_TO_XYZ_MATRIX,
                                       XYZ_TO_BETA_RGB_MATRIX,
                                       BETA_RGB_TRANSFER_FUNCTION,
                                       BETA_RGB_INVERSE_TRANSFER_FUNCTION)
