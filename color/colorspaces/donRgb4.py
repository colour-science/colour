#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**donRgb4.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *Don RGB 4* colorspace.

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
		   "DON_RGB_4_PRIMARIES",
		   "DON_RGB_4_WHITEPOINT",
		   "DON_RGB_4_TO_XYZ_MATRIX",
		   "XYZ_TO_DON_RGB_4_MATRIX",
		   "DON_RGB_4_TRANSFER_FUNCTION",
		   "DON_RGB_4_INVERSE_TRANSFER_FUNCTION",
		   "DON_RGB_4_COLORSPACE"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#*** *Don RGB 4*
#**********************************************************************************************************************
# http://www.hutchcolor.com/profiles/DonRGB4.zip
DON_RGB_4_PRIMARIES = numpy.matrix([0.69612068965517238, 0.29956896551724138,
									0.21468298109010012, 0.7652947719688542,
									0.12993762993762992, 0.035343035343035345]).reshape((3, 2))

DON_RGB_4_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

DON_RGB_4_TO_XYZ_MATRIX = color.derivation.getNormalizedPrimaryMatrix(DON_RGB_4_PRIMARIES, DON_RGB_4_WHITEPOINT)

XYZ_TO_DON_RGB_4_MATRIX = DON_RGB_4_TO_XYZ_MATRIX.getI()

def __donRgb4TransferFunction(RGB):
	"""
	Defines the *Don RGB 4* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __donRgb4InverseTransferFunction(RGB):
	"""
	Defines the *Don RGB 4* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

DON_RGB_4_TRANSFER_FUNCTION = __donRgb4TransferFunction

DON_RGB_4_INVERSE_TRANSFER_FUNCTION = __donRgb4InverseTransferFunction

DON_RGB_4_COLORSPACE = Colorspace("Don RGB 4",
								  DON_RGB_4_PRIMARIES,
								  DON_RGB_4_WHITEPOINT,
								  DON_RGB_4_TO_XYZ_MATRIX,
								  XYZ_TO_DON_RGB_4_MATRIX,
								  DON_RGB_4_TRANSFER_FUNCTION,
								  DON_RGB_4_INVERSE_TRANSFER_FUNCTION)