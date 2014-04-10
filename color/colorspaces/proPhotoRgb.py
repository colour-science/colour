#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**prophotoRgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *ProPhoto RGB* colorspace.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***    External imports.
#**********************************************************************************************************************
import numpy

#**********************************************************************************************************************
#***	Internal Imports.
#**********************************************************************************************************************
import color.exceptions
import color.illuminants
import color.verbose
from color.colorspaces.colorspace import Colorspace

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
		   "PROPHOTO_RGB_PRIMARIES",
		   "PROPHOTO_RGB_WHITEPOINT",
		   "PROPHOTO_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_PROPHOTO_RGB_MATRIX",
		   "PROPHOTO_RGB_TRANSFER_FUNCTION",
		   "PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION",
		   "PROPHOTO_RGB_COLORSPACE"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#*** *ProPhoto RGB*
#**********************************************************************************************************************
# http://www.color.org/ROMMRGB.pdf
PROPHOTO_RGB_PRIMARIES = numpy.matrix([0.7347, 0.2653,
									   0.1596, 0.8404,
									   0.0366, 0.0001]).reshape((3, 2))

PROPHOTO_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

PROPHOTO_RGB_TO_XYZ_MATRIX = numpy.matrix([7.97667235e-01, 1.35192231e-01, 3.13525290e-02,
										   2.88037454e-01, 7.11876883e-01, 8.56626476e-05,
										   0.00000000e+00, 0.00000000e+00, 8.25188285e-01]).reshape((3, 3))

XYZ_TO_PROPHOTO_RGB_MATRIX = PROPHOTO_RGB_TO_XYZ_MATRIX.getI()

def __prophotoRgbTransferFunction(RGB):
	"""
	Defines the *Prophoto RGB* colorspace transfer function.

	Reference: http://www.color.org/ROMMRGB.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: 0.003473 + 0.0622829 * x if x < 0.03125 else 0.003473 + 0.996527 * x ** ( 1 / 1.8 ),
			  numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __prophotoRgbTransferFunction(RGB):
	"""
	Defines the *Prophoto RGB* colorspace transfer function.

	Reference: http://www.color.org/ROMMRGB.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x * 16 if x < 0.001953 else x ** ( 1 / 1.8), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __prophotoRgbInverseTransferFunction(RGB):
	"""
	Defines the *Prophoto RGB* colorspace inverse transfer function.

	Reference: http://www.color.org/ROMMRGB.pdf

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x / 16 if x < 0.001953 else x ** 1.8, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

PROPHOTO_RGB_TRANSFER_FUNCTION = __prophotoRgbTransferFunction

PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION = __prophotoRgbInverseTransferFunction

PROPHOTO_RGB_COLORSPACE = Colorspace("ProPhoto RGB",
									 PROPHOTO_RGB_PRIMARIES,
									 PROPHOTO_RGB_WHITEPOINT,
									 PROPHOTO_RGB_TO_XYZ_MATRIX,
									 XYZ_TO_PROPHOTO_RGB_MATRIX,
									 PROPHOTO_RGB_TRANSFER_FUNCTION,
									 PROPHOTO_RGB_INVERSE_TRANSFER_FUNCTION)
