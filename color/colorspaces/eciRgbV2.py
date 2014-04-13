#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**eciRgbV2.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *ECI RGB v2* colorspace.

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
		   "ECI_RGB_V2_PRIMARIES",
		   "ECI_RGB_V2_WHITEPOINT",
		   "ECI_RGB_V2_TO_XYZ_MATRIX",
		   "XYZ_TO_ECI_RGB_V2_MATRIX",
		   "ECI_RGB_V2_TRANSFER_FUNCTION",
		   "ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION",
		   "ECI_RGB_V2_COLORSPACE"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#*** *ECI RGB v2*
#**********************************************************************************************************************
# http://www.eci.org/_media/downloads/icc_profiles_from_eci/ecirgbv20.zip
ECI_RGB_V2_PRIMARIES = numpy.matrix([0.67010309278350522, 0.32989690721649484,
									 0.20990566037735847, 0.70990566037735836,
									 0.14006179196704427, 0.080329557157569509]).reshape((3, 2))

ECI_RGB_V2_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

ECI_RGB_V2_TO_XYZ_MATRIX = color.derivation.getNormalizedPrimaryMatrix(ECI_RGB_V2_PRIMARIES, ECI_RGB_V2_WHITEPOINT)

XYZ_TO_ECI_RGB_V2_MATRIX = ECI_RGB_V2_TO_XYZ_MATRIX.getI()

def __eciRgbV2TransferFunction(RGB):
	"""
	Defines the *ECI RGB v2* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	# TODO: *ECI RGB v2* TRC should be *L\**.
	LOGGER.warning("!> {0} | 'ECI RGB v2' colorspace transfer function is not implemented!".format(__name__))
	RGB = map(lambda x: x ** (1 / 1.8), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __eciRgbV2InverseTransferFunction(RGB):
	"""
	Defines the *ECI RGB v2* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	# TODO: *ECI RGB v2* TRC should be inverse *L\**.
	LOGGER.warning("!> {0} | 'ECI RGB v2' colorspace inverse transfer function is not implemented!".format(__name__))
	RGB = map(lambda x: x ** 1.8, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

ECI_RGB_V2_TRANSFER_FUNCTION = __eciRgbV2TransferFunction

ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION = __eciRgbV2InverseTransferFunction

ECI_RGB_V2_COLORSPACE = Colorspace("ECI RGB v2",
								   ECI_RGB_V2_PRIMARIES,
								   ECI_RGB_V2_WHITEPOINT,
								   ECI_RGB_V2_TO_XYZ_MATRIX,
								   XYZ_TO_ECI_RGB_V2_MATRIX,
								   ECI_RGB_V2_TRANSFER_FUNCTION,
								   ECI_RGB_V2_INVERSE_TRANSFER_FUNCTION)