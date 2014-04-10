#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cieRgb.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *CIE RGB* colorspace.

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
		   "CIE_RGB_PRIMARIES",
		   "CIE_RGB_WHITEPOINT",
		   "CIE_RGB_TO_XYZ_MATRIX",
		   "XYZ_TO_CIE_RGB_MATRIX",
		   "CIE_RGB_TRANSFER_FUNCTION",
		   "CIE_RGB_INVERSE_TRANSFER_FUNCTION",
		   "CIE_RGB_COLORSPACE"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#*** *CIE RGB*
#**********************************************************************************************************************
# http://en.wikipedia.org/wiki/CIE_1931_color_space#Construction_of_the_CIE_XYZ_color_space_from_the_Wright.E2.80.93Guild_data
CIE_RGB_PRIMARIES = numpy.matrix([1., 0.,
								  0., 1.,
								  0., 0.]).reshape((3, 2))

CIE_RGB_WHITEPOINT = color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("E")

CIE_RGB_TO_XYZ_MATRIX = 1 / 0.17697 * numpy.matrix([0.49, 0.31, 0.20,
													0.17697, 0.81240, 0.01063,
													0.00, 0.01, 0.99]).reshape((3, 3))

XYZ_TO_CIE_RGB_MATRIX = CIE_RGB_TO_XYZ_MATRIX.getI()

CIE_RGB_TRANSFER_FUNCTION = lambda x: x

CIE_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x

CIE_RGB_COLORSPACE = Colorspace("CIE RGB",
								CIE_RGB_PRIMARIES,
								CIE_RGB_WHITEPOINT,
								CIE_RGB_TO_XYZ_MATRIX,
								XYZ_TO_CIE_RGB_MATRIX,
								CIE_RGB_TRANSFER_FUNCTION,
								CIE_RGB_INVERSE_TRANSFER_FUNCTION)