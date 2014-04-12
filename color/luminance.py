#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**luminance.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *luminance* objects.

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
import color.verbose

#**********************************************************************************************************************
#***    Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
		   "getLuminanceEquation",
		   "getLuminance"]

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#***    Module classes and definitions.
#**********************************************************************************************************************
def getLuminanceEquation(primaries, whitepoint):
	"""
	Returns the *luminance equation* from given *primaries* and *whitepoint* matrices.

	Reference: http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf: 3.3.8

	Usage::

		>>> primaries = numpy.matrix([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]).reshape((3, 2))
		>>> whitepoint = (0.32168, 0.33767)
		>>> getLuminanceEquation(primaries, whitepoint)
		Y = 0.343966449765(R) + 0.728166096613(G) + -0.0721325463786(B)

	:param primaries: Primaries chromaticity coordinate matrix.
	:type primaries: Matrix (3x2)
	:param whitepoint: Illuminant / whitepoint chromaticity coordinates.
	:type whitepoint: tuple
	:return: Luminance equation.
	:rtype: unicode
	"""

	return "Y = {0}(R) + {1}(G) + {2}(B)".format(
		*numpy.ravel(color.derivation.getNormalizedPrimaryMatrix(primaries, whitepoint))[3:6])

def getLuminance(RGB, primaries, whitepoint):
	"""
	Returns the *luminance* of given *RGB* components from given *primaries* and *whitepoint* matrices.

	Usage::

		>>> RGB = numpy.matrix([0.5, 0.5, 0.5]).reshape((3, 1))
		>>> primaries = numpy.matrix([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]).reshape((3, 2))
		>>> whitepoint = (0.32168, 0.33767)
		>>> getLuminance(primaries, whitepoint)

	:param RGB: *RGB* chromaticity coordinate matrix.
	:type RGB: Matrix (3x1)
	:param primaries: Primaries chromaticity coordinate matrix.
	:type primaries: Matrix (3x2)
	:param whitepoint: Illuminant / whitepoint chromaticity coordinates.
	:type whitepoint: tuple
	:return: Luminance.
	:rtype: float
	"""

	R, G, B = numpy.ravel(RGB)
	X, Y, Z = numpy.ravel(color.derivation.getNormalizedPrimaryMatrix(primaries, whitepoint))[3:6]

	return X * R + Y * G + Z * B