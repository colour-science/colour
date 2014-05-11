#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**beta_rgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *Beta RGB* colorspace.

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
		   "BETA_RGB_PRIMARIES",
		   "BETA_RGB_WHITEPOINT",
		   "BETA_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_BETA_RGB_MATRIX",
		   "BETA_RGB_TRANSFER_FUNCTION",
		   "BETA_RGB_INVERSE_TRANSFER_FUNCTION",
		   "BETA_RGB_COLORSPACE"]

LOGGER = color.verbose.install_logger()

# http://www.brucelindbloom.com/WorkingSpaceInfo.html
BETA_RGB_PRIMARIES = numpy.matrix([0.6888, 0.3112,
								   0.1986, 0.7551,
								   0.1265, 0.0352]).reshape((3, 2))

BETA_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")

BETA_RGB_TO_XYZ_MATRIX = color.derivation.get_normalized_primary_matrix(BETA_RGB_PRIMARIES, BETA_RGB_WHITEPOINT)

XYZ_TO_BETA_RGB_MATRIX = BETA_RGB_TO_XYZ_MATRIX.getI()

def __beta_rgb_transfer_function(RGB):
	"""
	Defines the *Beta RGB* colorspace transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** (1 / 2.2), numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

def __beta_rgb_inverse_transfer_function(RGB):
	"""
	Defines the *Beta RGB* colorspace inverse transfer function.

	:param RGB: RGB Matrix.
	:type RGB: Matrix (3x1)
	:return: Companded RGB Matrix.
	:rtype: Matrix (3x1)
	"""

	RGB = map(lambda x: x ** 2.2, numpy.ravel(RGB))
	return numpy.matrix(RGB).reshape((3, 1))

BETA_RGB_TRANSFER_FUNCTION = __beta_rgb_transfer_function

BETA_RGB_INVERSE_TRANSFER_FUNCTION = __beta_rgb_inverse_transfer_function

BETA_RGB_COLORSPACE = Colorspace("Beta RGB",
								 BETA_RGB_PRIMARIES,
								 BETA_RGB_WHITEPOINT,
								 BETA_RGB_TO_XYZ_MATRIX,
								 XYZ_TO_BETA_RGB_MATRIX,
								 BETA_RGB_TRANSFER_FUNCTION,
								 BETA_RGB_INVERSE_TRANSFER_FUNCTION)