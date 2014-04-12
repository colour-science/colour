#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**testsLuminance.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines units tests for :mod:`color.luminance` module.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	External imports.
#**********************************************************************************************************************
import numpy
import re
import sys

if sys.version_info[:2] <= (2, 6):
	import unittest2 as unittest
else:
	import unittest

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
import color.luminance

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["GetLuminanceEquationTestCase"]

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
class GetLuminanceEquationTestCase(unittest.TestCase):
	"""
	Defines :func:`color.luminance.getLuminanceEquation` definition units tests methods.
	"""

	def testGetLuminanceEquation(self):
		"""
		Tests :func:`color.luminance.getLuminanceEquation` definition.
		"""

		self.assertIsInstance(color.luminance.getLuminanceEquation(
			numpy.matrix([0.73470, 0.26530,
						  0.00000, 1.00000,
						  0.00010, -0.07700]).reshape(
				(3, 2)),
			(0.32168, 0.33767)), unicode)

		self.assertTrue(re.match(
			r"Y\s?=\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(R\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(G\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(B\)",
			color.luminance.getLuminanceEquation(numpy.matrix([0.73470, 0.26530,
															   0.00000, 1.00000,
															   0.00010, -0.07700]).reshape((3, 2)),
												 (0.32168, 0.33767))))

class GetLuminanceTestCase(unittest.TestCase):
	"""
	Defines :func:`color.luminance.getLuminance` definition units tests methods.
	"""

	def testGetLuminanceEquation(self):
		"""
		Tests :func:`color.luminance.getLuminance` definition.
		"""

		self.assertAlmostEqual(color.luminance.getLuminance(numpy.matrix([50., 50., 50.]),
															numpy.matrix([0.73470, 0.26530,
																		  0.00000, 1.00000,
																		  0.00010, -0.07700]).reshape(
																(3, 2)),
															(0.32168, 0.33767)),
							   50.,
							   places=7)

		self.assertAlmostEqual(color.luminance.getLuminance(numpy.matrix([74.6, 16.1, 100.]),
															numpy.matrix([0.73470, 0.26530,
																		  0.00000, 1.00000,
																		  0.00010, -0.07700]).reshape(
																(3, 2)),
															(0.32168, 0.33767)),
							   30.1701166701,
							   places=7)

		self.assertAlmostEqual(color.luminance.getLuminance(numpy.matrix([40.6, 4.2, 67.4]),
															numpy.matrix([0.73470, 0.26530,
																		  0.00000, 1.00000,
																		  0.00010, -0.07700]).reshape(
																(3, 2)),
															(0.32168, 0.33767)),
							   12.1616018403,
							   places=7)

if __name__ == "__main__":
	unittest.main()
