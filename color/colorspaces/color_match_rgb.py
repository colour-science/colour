#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**color_match_rgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *ColorMatch RGB* colorspace.

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
		   "COLOR_MATCH_RGB_PRIMARIES",
		   "COLOR_MATCH_RGB_WHITEPOINT",
		   "COLOR_MATCH_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_COLOR_MATCH_RGB_MATRIX",
		   "COLOR_MATCH_RGB_TRANSFER_FUNCTION",
		   "COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION",
		   "COLOR_MATCH_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

#**********************************************************************************************************************
#*** *ColorMatch RGB*
#**********************************************************************************************************************
# http://www.brucelindbloom.com/WorkingSpaceInfo.html
COLOR_MATCH_RGB_PRIMARIES = numpy.matrix([0.6300, 0.3400,
										  0.2950, 0.6050,
										  0.1500, 0.0750]).reshape((3, 2))

COLOR_MATCH_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

COLOR_MATCH_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(COLOR_MATCH_RGB_PRIMARIES,
																			COLOR_MATCH_RGB_WHITEPOINT)

XYZ_TO_COLOR_MATCH_RGB_MATRIX = COLOR_MATCH_RGB_TO_XYZ_MATRIX.getI()

def __color_match_rgb_transfer_function(RGB):
	"""
	Defines the *ColorMatch RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 1.8), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __color_match_rgb_inverse_transfer_function(RGB):
	"""
	Defines the *ColorMatch RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 1.8, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

COLOR_MATCH_RGB_TRANSFER_FUNCTION = __color_match_rgb_transfer_function

COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION = __color_match_rgb_inverse_transfer_function

COLOR_MATCH_RGB_COLORSPACE = Colorspace("ColorMatch RGB",
										COLOR_MATCH_RGB_PRIMARIES,
										COLOR_MATCH_RGB_WHITEPOINT,
										COLOR_MATCH_RGB_TO_XYZ_MATRIX,
										XYZ_TO_COLOR_MATCH_RGB_MATRIX,
										COLOR_MATCH_RGB_TRANSFER_FUNCTION,
										COLOR_MATCH_RGB_INVERSE_TRANSFER_FUNCTION)