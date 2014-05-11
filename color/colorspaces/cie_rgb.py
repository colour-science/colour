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
		   "CIE_RGB_PRIMARIES",
		   "CIE_RGB_WHITEPOINT",
		   "CIE_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_CIE_RGB_MATRIX",
		   "CIE_RGB_TRANSFER_FUNCTION",
		   "CIE_RGB_INVERSE_TRANSFER_FUNCTION",
		   "CIE_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright.E2.80.93Guild_data
CIE_RGB_PRIMARIES = numpy.matrix([0.7350, 0.2650,
								  0.2740, 0.7170,
								  0.1670, 0.0090]).reshape((3, 2))

CIE_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("E")

CIE_RGB_TO_XYZ_MATRIX = 1 / 0.17697 * numpy.matrix([0.49, 0.31, 0.20,
													0.17697, 0.81240, 0.01063,
													0.00, 0.01, 0.99]).reshape((3, 3))

XYZ_TO_CIE_RGB_MATRIX = CIE_RGB_TO_XYZ_MATRIX.getI()

def __cie_rgb_transfer_function(RGB):
	"""
	Defines the *CIE RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __cie_rgb_inverse_transfer_function(RGB):
	"""
	Defines the *CIE RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

CIE_RGB_TRANSFER_FUNCTION = __cie_rgb_transfer_function

CIE_RGB_INVERSE_TRANSFER_FUNCTION = __cie_rgb_inverse_transfer_function

CIE_RGB_COLORSPACE = Colorspace("CIE RGB",
								CIE_RGB_PRIMARIES,
								CIE_RGB_WHITEPOINT,
								CIE_RGB_TO_XYZ_MATRIX,
								XYZ_TO_CIE_RGB_MATRIX,
								CIE_RGB_TRANSFER_FUNCTION,
								CIE_RGB_INVERSE_TRANSFER_FUNCTION)