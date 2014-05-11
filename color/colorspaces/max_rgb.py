#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**max_rgb.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *Max RGB* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.derivation
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
           "MAX_RGB_PRIMARIES",
           "MAX_RGB_WHITEPOINT",
           "MAX_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_MAX_RGB_MATRIX",
           "MAX_RGB_TRANSFER_FUNCTION",
           "MAX_RGB_INVERSE_TRANSFER_FUNCTION",
           "MAX_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://www.hutchcolor.com/profiles/MaxRGB.zip
MAX_RGB_PRIMARIES = numpy.matrix([0.73413379, 0.26586621,
                                  0.10039113, 0.89960887,
                                  0.03621495, 0.]).reshape((3, 2))

MAX_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

MAX_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(MAX_RGB_PRIMARIES, MAX_RGB_WHITEPOINT)

XYZ_TO_MAX_RGB_MATRIX = MAX_RGB_TO_XYZ_MATRIX.getI()

def __max_rgb_transfer_function(RGB):
    """
    Defines the *Max RGB* colorspace transfer function.

    :param RGB: RGB Matrix.
    :type RGB: Matrix (3x1)
    :return: Companded RGB Matrix.
    :rtype: Matrix (3x1)
    """

    RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
    return numpy.matrix(RGB).reshape((3, 1))

def __max_rgb_inverse_transfer_function(RGB):
    """
    Defines the *Max RGB* colorspace inverse transfer function.

    :param RGB: RGB Matrix.
    :type RGB: Matrix (3x1)
    :return: Companded RGB Matrix.
    :rtype: Matrix (3x1)
    """

    RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
    return numpy.matrix(RGB).reshape((3, 1))

MAX_RGB_TRANSFER_FUNCTION = __max_rgb_transfer_function

MAX_RGB_INVERSE_TRANSFER_FUNCTION = __max_rgb_inverse_transfer_function

MAX_RGB_COLORSPACE = Colorspace("Max RGB",
                                MAX_RGB_PRIMARIES,
                                MAX_RGB_WHITEPOINT,
                                MAX_RGB_TO_XYZ_MATRIX,
                                XYZ_TO_MAX_RGB_MATRIX,
                                MAX_RGB_TRANSFER_FUNCTION,
                                MAX_RGB_INVERSE_TRANSFER_FUNCTION)
