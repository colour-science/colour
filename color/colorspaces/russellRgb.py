#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**russellRgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *Russell RGB* colorspace.

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
import color.derivation
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
		   "RUSSELL_RGB_PRIMARIES",
		   "RUSSELL_RGB_WHITEPOINT",
		   "RUSSELL_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_RUSSELL_RGB_MATRIX",
		   "RUSSELL_RGB_TRANSFER_FUNCTION",
		   "RUSSELL_RGB_INVERSE_TRANSFER_FUNCTION",
		   "RUSSELL_RGB_COLORSPACE"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#*** *Russell RGB*
#**********************************************************************************************************************
# http://www.russellcottrell.com/photo/RussellRGB.htm
RUSSELL_RGB_PRIMARIES = numpy.matrix([0.6900, 0.3100,
									  0.1800, 0.7700,
									  0.1000, 0.0200]).reshape((3, 2))

RUSSELL_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D55")

RUSSELL_RGB_TO_XYZ_MATRIX = color.derivation.getNormalizedPrimaryMatrix(RUSSELL_RGB_PRIMARIES, RUSSELL_RGB_WHITEPOINT)

XYZ_TO_RUSSELL_RGB_MATRIX = RUSSELL_RGB_TO_XYZ_MATRIX.getI()

def __russellRgbTransferFunction(RGB):
	"""
	Defines the *Russell RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __russellRgbInverseTransferFunction(RGB):
	"""
	Defines the *Russell RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

RUSSELL_RGB_TRANSFER_FUNCTION = __russellRgbTransferFunction

RUSSELL_RGB_INVERSE_TRANSFER_FUNCTION = __russellRgbInverseTransferFunction

RUSSELL_RGB_COLORSPACE = Colorspace("Russell RGB",
									RUSSELL_RGB_PRIMARIES,
									RUSSELL_RGB_WHITEPOINT,
									RUSSELL_RGB_TO_XYZ_MATRIX,
									XYZ_TO_RUSSELL_RGB_MATRIX,
									RUSSELL_RGB_TRANSFER_FUNCTION,
									RUSSELL_RGB_INVERSE_TRANSFER_FUNCTION)