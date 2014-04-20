#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**adobe_wide_gamut_rgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *Adobe Wide Gamut RGB* colorspace.

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
		   "ADOBE_WIDE_GAMUT_RGB_PRIMARIES",
		   "ADOBE_WIDE_GAMUT_RGB_WHITEPOINT",
		   "ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX",
		   "ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION",
		   "ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION",
		   "ADOBE_WIDE_GAMUT_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

#**********************************************************************************************************************
#*** *Adobe Wide Gamut RGB*
#**********************************************************************************************************************
# http://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space
ADOBE_WIDE_GAMUT_RGB_PRIMARIES = numpy.matrix([0.7347, 0.2653,
											   0.1152, 0.8264,
											   0.1566, 0.0177]).reshape((3, 2))

ADOBE_WIDE_GAMUT_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
																				 ADOBE_WIDE_GAMUT_RGB_WHITEPOINT)

XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX = ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX.getI()

def __adobe_wide_gamut_rgb_transfer_function(RGB):
	"""
	Defines the *Adobe Wide Gamut RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / (563. / 256.)), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __adobe_wide_gamut_rgb_inverse_transfer_function(RGB):
	"""
	Defines the *Adobe Wide Gamut RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (563. / 256.), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION = __adobe_wide_gamut_rgb_transfer_function

ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION = __adobe_wide_gamut_rgb_inverse_transfer_function

ADOBE_WIDE_GAMUT_RGB_COLORSPACE = Colorspace("Adobe Wide Gamut RGB",
											 ADOBE_WIDE_GAMUT_RGB_PRIMARIES,
											 ADOBE_WIDE_GAMUT_RGB_WHITEPOINT,
											 ADOBE_WIDE_GAMUT_RGB_TO_XYZ_MATRIX,
											 XYZ_TO_ADOBE_WIDE_GAMUT_RGB_MATRIX,
											 ADOBE_WIDE_GAMUT_RGB_TRANSFER_FUNCTION,
											 ADOBE_WIDE_GAMUT_RGB_INVERSE_TRANSFER_FUNCTION)