#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**ntsc_rgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *NTSC RGB* colorspace.

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
		   "NTSC_RGB_PRIMARIES",
		   "NTSC_RGB_WHITEPOINT",
		   "NTSC_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_NTSC_RGB_MATRIX",
		   "NTSC_RGB_TRANSFER_FUNCTION",
		   "NTSC_RGB_INVERSE_TRANSFER_FUNCTION",
		   "NTSC_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

#**********************************************************************************************************************
#*** *NTSC RGB*
#**********************************************************************************************************************
# http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf
NTSC_RGB_PRIMARIES = numpy.matrix([0.67, 0.33,
								   0.21, 0.71,
								   0.14, 0.08]).reshape((3, 2))

NTSC_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("C")

NTSC_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(NTSC_RGB_PRIMARIES, NTSC_RGB_WHITEPOINT)

XYZ_TO_NTSC_RGB_MATRIX = NTSC_RGB_TO_XYZ_MATRIX.getI()

def __ntsc_rgb_transfer_function(RGB):
	"""
	Defines the *NTSC RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __ntsc_rgb_inverse_transfer_function(RGB):
	"""
	Defines the *NTSC RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

NTSC_RGB_TRANSFER_FUNCTION = __ntsc_rgb_transfer_function

NTSC_RGB_INVERSE_TRANSFER_FUNCTION = __ntsc_rgb_inverse_transfer_function

NTSC_RGB_COLORSPACE = Colorspace("NTSC RGB",
								 NTSC_RGB_PRIMARIES,
								 NTSC_RGB_WHITEPOINT,
								 NTSC_RGB_TO_XYZ_MATRIX,
								 XYZ_TO_NTSC_RGB_MATRIX,
								 NTSC_RGB_TRANSFER_FUNCTION,
								 NTSC_RGB_INVERSE_TRANSFER_FUNCTION)