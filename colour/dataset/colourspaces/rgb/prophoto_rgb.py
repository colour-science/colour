# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**prophotoRgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *ProPhoto RGB* colourspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.dataset.illuminants.chromaticity_coordinates
from colour.computation.colourspaces.rgb.colourspace import Colourspace

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["PROPHOTO_RGB_PRIMARIES",
           "PROPHOTO_RGB_WHITEPOINT",
           "PROPHOTO_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_PROPHOTO_RGB_MATRIX",
           "PROPHOTO_RGB_TRANSFER_FUNCTION",
           "PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION",
           "PROPHOTO_RGB_COLOURSPACE"]


# http://www.color.org/ROMMRGB.pdf
PROPHOTO_RGB_PRIMARIES = numpy.array([0.7347, 0.2653,
                                      0.1596, 0.8404,
                                      0.0366, 0.0001]).reshape((3, 2))

PROPHOTO_RGB_WHITEPOINT = colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
    "CIE 1931 2 Degree Standard Observer").get("D50")

PROPHOTO_RGB_TO_XYZ_MATRIX = numpy.array([7.97667235e-01, 1.35192231e-01, 3.13525290e-02,
                                          2.88037454e-01, 7.11876883e-01, 8.56626476e-05,
                                          0.00000000e+00, 0.00000000e+00, 8.25188285e-01]).reshape((3, 3))

XYZ_TO_PROPHOTO_RGB_MATRIX = numpy.linalg.inv(PROPHOTO_RGB_TO_XYZ_MATRIX)

PROPHOTO_RGB_TRANSFER_FUNCTION = lambda x: x * 16 if x < 0.001953 else x ** (1 / 1.8)

PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x / 16 if x < 0.001953 else x ** 1.8

PROPHOTO_RGB_COLOURSPACE = Colourspace("ProPhoto RGB",
                                       PROPHOTO_RGB_PRIMARIES,
                                       PROPHOTO_RGB_WHITEPOINT,
                                       PROPHOTO_RGB_TO_XYZ_MATRIX,
                                       XYZ_TO_PROPHOTO_RGB_MATRIX,
                                       PROPHOTO_RGB_TRANSFER_FUNCTION,
                                       PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION)
