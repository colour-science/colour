#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**pal_secam_rgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *Pal/Secam RGB* colorspace.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.derivation
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
		   "PAL_SECAM_RGB_PRIMARIES",
		   "PAL_SECAM_RGB_WHITEPOINT",
		   "PAL_SECAM_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_PAL_SECAM_RGB_MATRIX",
		   "PAL_SECAM_RGB_TRANSFER_FUNCTION",
		   "PAL_SECAM_RGB_INVERSE_TRANSFER_FUNCTION",
		   "PAL_SECAM_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf
PAL_SECAM_RGB_PRIMARIES = numpy.matrix([0.64, 0.33,
										0.29, 0.60,
										0.15, 0.06]).reshape((3, 2))

PAL_SECAM_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D65")

PAL_SECAM_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(PAL_SECAM_RGB_PRIMARIES,
																		  PAL_SECAM_RGB_WHITEPOINT)

XYZ_TO_PAL_SECAM_RGB_MATRIX = PAL_SECAM_RGB_TO_XYZ_MATRIX.getI()

def __pal_secam_rgb_transfer_function(RGB):
	"""
	Defines the *Pal/Secam RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.8), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __pal_secam_rgb_inverse_transfer_function(RGB):
	"""
	Defines the *Pal/Secam RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.8, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

PAL_SECAM_RGB_TRANSFER_FUNCTION = __pal_secam_rgb_transfer_function

PAL_SECAM_RGB_INVERSE_TRANSFER_FUNCTION = __pal_secam_rgb_inverse_transfer_function

PAL_SECAM_RGB_COLORSPACE = Colorspace("Pal/Secam RGB",
									  PAL_SECAM_RGB_PRIMARIES,
									  PAL_SECAM_RGB_WHITEPOINT,
									  PAL_SECAM_RGB_TO_XYZ_MATRIX,
									  XYZ_TO_PAL_SECAM_RGB_MATRIX,
									  PAL_SECAM_RGB_TRANSFER_FUNCTION,
									  PAL_SECAM_RGB_INVERSE_TRANSFER_FUNCTION)