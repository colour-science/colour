#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**smptec_rgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *SMPTE-C RGB* colorspace.

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
		   "SMPTE_C_RGB_PRIMARIES",
		   "SMPTE_C_RGB_WHITEPOINT",
		   "SMPTE_C_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_SMPTE_C_RGB_MATRIX",
		   "SMPTE_C_RGB_TRANSFER_FUNCTION",
		   "SMPTE_C_RGB_INVERSE_TRANSFER_FUNCTION",
		   "SMPTE_C_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

#**********************************************************************************************************************
#*** *SMPTE-C RGB*
#**********************************************************************************************************************
# http://standards.smpte.org/content/978-1-61482-164-9/rp-145-2004/SEC1.body.pdf
SMPTE_C_RGB_PRIMARIES = numpy.matrix([0.630, 0.340,
									  0.310, 0.595,
									  0.155, 0.070]).reshape((3, 2))

SMPTE_C_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D65")

SMPTE_C_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(SMPTE_C_RGB_PRIMARIES, SMPTE_C_RGB_WHITEPOINT)

XYZ_TO_SMPTE_C_RGB_MATRIX = SMPTE_C_RGB_TO_XYZ_MATRIX.getI()

def __smptec_rgb_transfer_function(RGB):
	"""
	Defines the *SMPTE-C RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __smptec_rgb_inverse_transfer_function(RGB):
	"""
	Defines the *SMPTE-C RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

SMPTE_C_RGB_TRANSFER_FUNCTION = __smptec_rgb_transfer_function

SMPTE_C_RGB_INVERSE_TRANSFER_FUNCTION = __smptec_rgb_inverse_transfer_function

SMPTE_C_RGB_COLORSPACE = Colorspace("SMPTE-C RGB",
									SMPTE_C_RGB_PRIMARIES,
									SMPTE_C_RGB_WHITEPOINT,
									SMPTE_C_RGB_TO_XYZ_MATRIX,
									XYZ_TO_SMPTE_C_RGB_MATRIX,
									SMPTE_C_RGB_TRANSFER_FUNCTION,
									SMPTE_C_RGB_INVERSE_TRANSFER_FUNCTION)