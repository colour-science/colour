#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_lightness.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines units tests for :mod:`color.lightness` module.

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
import color.lightness

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["GetLuminanceEquationTestCase",
		   "GetLuminanceTestCase",
           "Luminance1943TestCase",
           "Luminance1976TestCase",
           "MunsellValue1920TestCase",
           "MunsellValue1933TestCase",
           "MunsellValue1943TestCase",
           "MunsellValue1944TestCase",
           "MunsellValue1955TestCase",
           "Lightness1958TestCase",
           "Lightness1964TestCase",
           "Lightness1976TestCase"]

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
class GetLuminanceEquationTestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.get_luminance_equation` definition units tests methods.
	"""

	def test_get_luminance_equation(self):
		"""
		Tests :func:`color.lightness.get_luminance_equation` definition.
		"""

		self.assertIsInstance(color.lightness.get_luminance_equation(
			numpy.matrix([0.73470, 0.26530,
						  0.00000, 1.00000,
						  0.00010, -0.07700]).reshape(
				(3, 2)),
			(0.32168, 0.33767)), unicode)

		self.assertTrue(re.match(
			r"Y\s?=\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(R\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(G\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(B\)",
			color.lightness.get_luminance_equation(numpy.matrix([0.73470, 0.26530,
															   0.00000, 1.00000,
															   0.00010, -0.07700]).reshape((3, 2)),
												 (0.32168, 0.33767))))

class GetLuminanceTestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.get_luminance` definition units tests methods.
	"""

	def test_get_luminance(self):
		"""
		Tests :func:`color.lightness.get_luminance` definition.
		"""

		self.assertAlmostEqual(color.lightness.get_luminance(numpy.matrix([50., 50., 50.]),
															numpy.matrix([0.73470, 0.26530,
																		  0.00000, 1.00000,
																		  0.00010, -0.07700]).reshape(
																(3, 2)),
															(0.32168, 0.33767)),
							   50.,
							   places=7)

		self.assertAlmostEqual(color.lightness.get_luminance(numpy.matrix([74.6, 16.1, 100.]),
															numpy.matrix([0.73470, 0.26530,
																		  0.00000, 1.00000,
																		  0.00010, -0.07700]).reshape(
																(3, 2)),
															(0.32168, 0.33767)),
							   30.1701166701,
							   places=7)

		self.assertAlmostEqual(color.lightness.get_luminance(numpy.matrix([40.6, 4.2, 67.4]),
															numpy.matrix([0.73470, 0.26530,
																		  0.00000, 1.00000,
																		  0.00010, -0.07700]).reshape(
																(3, 2)),
															(0.32168, 0.33767)),
							   12.1616018403,
							   places=7)

class Luminance1943TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.luminance_1943` definition units tests methods.
	"""

	def test_luminance_1943(self):
		"""
		Tests :func:`color.lightness.luminance_1943` definition.
		"""

		self.assertAlmostEqual(color.lightness.luminance_1943(3.74629715382), 10.4089874577, places=7)
		self.assertAlmostEqual(color.lightness.luminance_1943(8.64728711385), 71.3174801757, places=7)
		self.assertAlmostEqual(color.lightness.luminance_1943(1.52569021578), 2.06998750444, places=7)

class Luminance1976TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.luminance_1976` definition units tests methods.
	"""

	def test_luminance_1976(self):
		"""
		Tests :func:`color.lightness.luminance_1976` definition.
		"""

		self.assertAlmostEqual(color.lightness.luminance_1976(37.9856290977, 100.), 10.08, places=7)
		self.assertAlmostEqual(color.lightness.luminance_1976(80.0444155585, 100.), 56.76, places=7)
		self.assertAlmostEqual(color.lightness.luminance_1976(99.3467279026, 100.), 98.32, places=7)

class MunsellValue1920TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.munsell_value_1920` definition units tests methods.
	"""

	def test_munsell_value_1920(self):
		"""
		Tests :func:`color.lightness.munsell_value_1920` definition.
		"""

		self.assertAlmostEqual(color.lightness.munsell_value_1920(10.08), 3.17490157328, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1920(56.76), 7.53392328073, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1920(98.32), 9.91564420499, places=7)

class MunsellValue1933TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.munsell_value_1933` definition units tests methods.
	"""

	def test_munsell_value_1933(self):
		"""
		Tests :func:`color.lightness.munsell_value_1933` definition.
		"""

		self.assertAlmostEqual(color.lightness.munsell_value_1933(10.08), 3.79183555086, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1933(56.76), 8.27013181776, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1933(98.32), 9.95457710587, places=7)

class MunsellValue1943TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.munsell_value_1943` definition units tests methods.
	"""

	def test_munsell_value_1943(self):
		"""
		Tests :func:`color.lightness.munsell_value_1943` definition.
		"""

		self.assertAlmostEqual(color.lightness.munsell_value_1943(10.08), 3.74629715382, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1943(56.76), 7.8225814259, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1943(98.32), 9.88538236116, places=7)

class MunsellValue1944TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.munsell_value_1944` definition units tests methods.
	"""

	def test_munsell_value_1944(self):
		"""
		Tests :func:`color.lightness.munsell_value_1944` definition.
		"""

		self.assertAlmostEqual(color.lightness.munsell_value_1944(10.08), 3.68650805994, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1944(56.76), 7.89881184275, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1944(98.32), 9.85197100995, places=7)

class MunsellValue1955TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.munsell_value_1955` definition units tests methods.
	"""

	def test_munsell_value_1955(self):
		"""
		Tests :func:`color.lightness.munsell_value_1955` definition.
		"""

		self.assertAlmostEqual(color.lightness.munsell_value_1955(10.08), 3.69528622419, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1955(56.76), 7.84875137062, places=7)
		self.assertAlmostEqual(color.lightness.munsell_value_1955(98.32), 9.75492813681, places=7)

class Lightness1958TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.lightness_1958` definition units tests methods.
	"""

	def test_lightness_1958(self):
		"""
		Tests :func:`color.lightness.lightness_1958` definition.
		"""

		self.assertAlmostEqual(color.lightness.lightness_1958(10.08), 36.2505626458, places=7)
		self.assertAlmostEqual(color.lightness.lightness_1958(56.76), 78.8117999039, places=7)
		self.assertAlmostEqual(color.lightness.lightness_1958(98.32), 98.3447052593, places=7)

class Lightness1964TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.lightness_1964` definition units tests methods.
	"""

	def test_lightness_1964(self):
		"""
		Tests :func:`color.lightness.lightness_1964` definition.
		"""

		self.assertAlmostEqual(color.lightness.lightness_1964(10.08), 37.0041149128, places=7)
		self.assertAlmostEqual(color.lightness.lightness_1964(56.76), 79.0773031869, places=7)
		self.assertAlmostEqual(color.lightness.lightness_1964(98.32), 98.3862250488, places=7)

class Lightness1976TestCase(unittest.TestCase):
	"""
	Defines :func:`color.lightness.lightness_1976` definition units tests methods.
	"""

	def test_lightness_1976(self):
		"""
		Tests :func:`color.lightness.lightness_1976` definition.
		"""

		self.assertAlmostEqual(color.lightness.lightness_1976(10.08, 100.), 37.9856290977, places=7)
		self.assertAlmostEqual(color.lightness.lightness_1976(56.76, 100.), 80.0444155585, places=7)
		self.assertAlmostEqual(color.lightness.lightness_1976(98.32, 100.), 99.3467279026, places=7)

if __name__ == "__main__":
	unittest.main()
