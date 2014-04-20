#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_difference.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines units tests for :mod:`color.difference` module.

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
import sys

if sys.version_info[:2] <= (2, 6):
	import unittest2 as unittest
else:
	import unittest

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
import color.difference

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Delta_E_CIE_1976TestCase",
		   "Delta_E_CIE_1994TestCase",
		   "Delta_E_CIE_2000TestCase",
		   "Delta_E_CMCTestCase"]

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
class Delta_E_CIE_1976TestCase(unittest.TestCase):
	"""
	Defines :func:`color.difference.delta_E_CIE_1976` definition units tests methods.
	"""

	def test_delta_E_CIE_1976(self):
		"""
		Tests :func:`color.difference.delta_E_CIE_1976` definition.
		"""

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1976(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))),
			451.713301974,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1976(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 74.05216981, 276.45318193]).reshape((3, 1))),
			52.6498611564,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1976(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 8.32281957, -73.58297716]).reshape((3, 1))),
			346.064891718,
			places=7)

class Delta_E_CIE_1994TestCase(unittest.TestCase):
	"""
	Defines :func:`color.difference.delta_E_CIE_1994` definition units tests methods.
	"""

	def test_delta_E_CIE_1994(self):
		"""
		Tests :func:`color.difference.delta_E_CIE_1994` definition.
		"""

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1994(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))),
			88.3355530575,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1994(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 74.05216981, 276.45318193]).reshape((3, 1))),
			10.61265789,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1994(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 8.32281957, -73.58297716]).reshape((3, 1))),
			60.3686872611,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1994(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1)),
											  textiles=False),
			83.7792255009,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1994(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 74.05216981, 276.45318193]).reshape((3, 1)),
											  textiles=False),
			10.0539319546,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_1994(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 8.32281957, -73.58297716]).reshape((3, 1)),
											  textiles=False),
			57.5354537067,
			places=7)

class Delta_E_CIE_2000TestCase(unittest.TestCase):
	"""
	Defines :func:`color.difference.delta_E_CIE_2000` definition units tests methods.
	"""

	def test_delta_E_CIE_2000(self):
		"""
		Tests :func:`color.difference.delta_E_CIE_2000` definition.
		"""

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_2000(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))),
			94.0356490267,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_2000(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 74.05216981, 276.45318193]).reshape((3, 1))),
			14.8790641937,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CIE_2000(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
											  numpy.matrix([100., 8.32281957, -73.58297716]).reshape((3, 1))),
			68.2309487895,
			places=7)

class Delta_E_CMCTestCase(unittest.TestCase):
	"""
	Defines :func:`color.difference.delta_E_CMC` definition units tests methods.
	"""

	def test_delta_E_CMC(self):
		"""
		Tests :func:`color.difference.delta_E_CMC` definition.
		"""

		self.assertAlmostEqual(
			color.difference.delta_E_CMC(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
										 numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1))),
			172.704771287,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CMC(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
										 numpy.matrix([100., 74.05216981, 276.45318193]).reshape((3, 1))),
			20.5973271674,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CMC(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
										 numpy.matrix([100., 8.32281957, -73.58297716]).reshape((3, 1))),
			121.718414791,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CMC(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
										 numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1)),
										 l=1.),
			172.704771287,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CMC(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
										 numpy.matrix([100., 74.05216981, 276.45318193]).reshape((3, 1)),
										 l=1.),
			20.5973271674,
			places=7)

		self.assertAlmostEqual(
			color.difference.delta_E_CMC(numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
										 numpy.matrix([100., 8.32281957, -73.58297716]).reshape((3, 1)),
										 l=1.),
			121.718414791,
			places=7)

if __name__ == "__main__":
	unittest.main()
