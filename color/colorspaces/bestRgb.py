#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**bestRgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *Best RGB* colorspace.

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
		   "BEST_RGB_PRIMARIES",
		   "BEST_RGB_WHITEPOINT",
		   "BEST_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_BEST_RGB_MATRIX",
		   "BEST_RGB_TRANSFER_FUNCTION",
		   "BEST_RGB_INVERSE_TRANSFER_FUNCTION",
		   "BEST_RGB_COLORSPACE"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#*** *Best RGB*
#**********************************************************************************************************************
# http://www.hutchcolor.com/profiles/BestRGB.zip
BEST_RGB_PRIMARIES = numpy.matrix([0.7347, 0.2653,
								   0.2150, 0.7750,
								   0.1300, 0.0350]).reshape((3, 2))

BEST_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

BEST_RGB_TO_XYZ_MATRIX = color.derivation.getNormalizedPrimaryMatrix(BEST_RGB_PRIMARIES, BEST_RGB_WHITEPOINT)

XYZ_TO_BEST_RGB_MATRIX = BEST_RGB_TO_XYZ_MATRIX.getI()

def __bestRgbTransferFunction(RGB):
	"""
	Defines the *Best RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __bestRgbInverseTransferFunction(RGB):
	"""
	Defines the *Best RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

BEST_RGB_TRANSFER_FUNCTION = __bestRgbTransferFunction

BEST_RGB_INVERSE_TRANSFER_FUNCTION = __bestRgbInverseTransferFunction

BEST_RGB_COLORSPACE = Colorspace("Best RGB",
								 BEST_RGB_PRIMARIES,
								 BEST_RGB_WHITEPOINT,
								 BEST_RGB_TO_XYZ_MATRIX,
								 XYZ_TO_BEST_RGB_MATRIX,
								 BEST_RGB_TRANSFER_FUNCTION,
								 BEST_RGB_INVERSE_TRANSFER_FUNCTION)