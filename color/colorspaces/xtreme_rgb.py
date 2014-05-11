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
           "XTREME_RGB_PRIMARIES",
           "XTREME_RGB_WHITEPOINT",
           "XTREME_RGB_TO_XYZ_MATRIX",
           "XYZ_TO_XTREME_RGB_MATRIX",
           "XTREME_RGB_TRANSFER_FUNCTION",
           "XTREME_RGB_INVERSE_TRANSFER_FUNCTION",
           "XTREME_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://www.hutchcolor.com/profiles/XtremeRGB.zip
XTREME_RGB_PRIMARIES = numpy.matrix([1., 0.,
                                     0., 1.,
                                     0., 0.]).reshape((3, 2))

XTREME_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

XTREME_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(XTREME_RGB_PRIMARIES, XTREME_RGB_WHITEPOINT)

XYZ_TO_XTREME_RGB_MATRIX = XTREME_RGB_TO_XYZ_MATRIX.getI()

def __xtreme_rgb_transfer_function(RGB):
    """
    Defines the *Xtreme RGB* colorspace transfer function.

    :param RGB: RGB Matrix.
    :type RGB: Matrix (3x1)
    :return: Companded RGB Matrix.
    :rtype: Matrix (3x1)
    """

    RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
    return numpy.matrix(RGB).reshape((3, 1))

def __xtreme_rgb_inverse_transfer_function(RGB):
    """
    Defines the *Xtreme RGB* colorspace inverse transfer function.

    :param RGB: RGB Matrix.
    :type RGB: Matrix (3x1)
    :return: Companded RGB Matrix.
    :rtype: Matrix (3x1)
    """

    RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
    return numpy.matrix(RGB).reshape((3, 1))

XTREME_RGB_TRANSFER_FUNCTION = __xtreme_rgb_transfer_function

XTREME_RGB_INVERSE_TRANSFER_FUNCTION = __xtreme_rgb_inverse_transfer_function

XTREME_RGB_COLORSPACE = Colorspace("Xtreme RGB",
                                   XTREME_RGB_PRIMARIES,
                                   XTREME_RGB_WHITEPOINT,
                                   XTREME_RGB_TO_XYZ_MATRIX,
                                   XYZ_TO_XTREME_RGB_MATRIX,
                                   XTREME_RGB_TRANSFER_FUNCTION,
                                   XTREME_RGB_INVERSE_TRANSFER_FUNCTION)
